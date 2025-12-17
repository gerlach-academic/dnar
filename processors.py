import math

import torch
from torch.nn import Linear, ModuleList, ReLU, Sequential
from torch.nn.functional import binary_cross_entropy_with_logits
from torch_geometric.utils import group_argsort, scatter, softmax

from configs import base_config
from generate_data import EDGE_MASK_ONE, MASK, NODE_MASK_ONE, NODE_POINTER, SPEC
from utils import from_binary_states, gumbel_softmax, node_pointer_loss, temp_by_step



class StatesEncoder(torch.nn.Module):
    def __init__(self, h, num_binary_states):
        super().__init__()
        self.emb = torch.nn.Embedding(2**num_binary_states, h)

    def forward(self, states):
        return self.emb(from_binary_states(states))


class SelectBest(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.h
        self.emb = torch.nn.Embedding(2 ** (config.num_node_states + 1), config.h)

    def forward(self, binary_states, scalars, index):
        states = 2 * from_binary_states(binary_states)
        group_with_reciever = torch.cat( #just a stacking of the states & (batch/edge) index
            [torch.unsqueeze(states, -1), torch.unsqueeze(index, -1)], dim=1
        ) # shape [n_nodes, n_indexes(=2)], each combination is a group
        _, group_index = torch.unique( #finds the unique groups based on the combined state, throws away the batch index, returns the unique group index per node
            group_with_reciever, sorted=False, return_inverse=True, dim=0
        )

        #1 for best value inside the group, 0 else
        best_in_group = gumbel_softmax( #dunno why it uses gumbel softmax here, not just argmax    
            -scalars.squeeze(), #chooses the minimum scalar
            group_index, 
            tau=0.0, 
            use_noise=False
        )

        state_with_best = states + best_in_group
        return self.emb(state_with_best.long())


class AttentionModule(torch.nn.Module):
    def __init__(self, config: base_config.Config):
        super().__init__()
        h = config.h
        self.h = h

        self.edge_states_encoder = StatesEncoder(config.h, config.num_edge_states)

        self.lin_query = Linear(h, h, bias=False)
        self.lin_key = Linear(h, h, bias=False)
        self.lin_value = Linear(h, h, bias=False)

        self.edge_key = Linear(h, h, bias=False)
        self.edge_value = Linear(h, h, bias=False)

        self.select_best_virtual = SelectBest(config)
        self.select_best_by_reciever = SelectBest(config)

        self.static_fts_encoder = StatesEncoder(h, 2)
        self.combine_fts = Linear(3 * h, h, bias=False)

        self.use_noise = config.use_noise
        self.temp = (
            config.processor_upper_t,
            config.processor_lower_t,
            config.num_iterations,
            config.temp_on_eval,
        )

    def forward(self, node_states, edge_states, scalars, batch, training_step):
        node_fts = self.select_best_from_virtual(node_states, scalars, batch)
        edge_fts = self.edge_states_encoder(edge_states)

        Q = self.lin_query(node_fts)
        K = self.lin_key(node_fts)
        V = self.lin_value(node_fts)

        edge_K, edge_V = self.combined_edge_KV(node_states, edge_fts, scalars, batch)

        message = self.compute_message(
            Q=Q,
            K=K,
            V=V,
            edge_K=edge_K,
            edge_V=edge_V,
            edge_index=batch.edge_index,
            training_step=training_step,
        )

        node_fts = node_fts + scatter(message, index=batch.edge_index[1])
        edge_fts = edge_fts + message
        return node_fts, edge_fts

    def compute_message(self, Q, K, V, edge_K, edge_V, edge_index, training_step):
        Q = Q[edge_index[1]]
        K = K[edge_index[0]] + edge_K
        V = V[edge_index[0]] + edge_V

        alpha = (Q * K).sum(dim=-1) / math.sqrt(self.h)

        tau = temp_by_step(training_step, *self.temp)
        use_noise = self.use_noise and training_step != -1

        alpha = gumbel_softmax(alpha, edge_index[1], tau=tau, use_noise=use_noise)

        return V * alpha.view(-1, 1)

    def compute_static_fts(self, scalars, batch):
        node_scalars = scalars[batch.edge_index[0] == batch.edge_index[1]]
        sender_s = node_scalars[batch.edge_index[0]]
        reciever_s = node_scalars[batch.edge_index[1]]

        #relaxation that are gigantically helpful inductive biases for easier inference
        #TODO: remove the relaxation and run thrice with bfs/dijkstra and see if the model can learn without the following features:
        rlx = scalars < reciever_s
        rlx_d = sender_s + scalars < reciever_s

        fts = torch.cat([rlx, rlx_d], dim=-1).long()
        return self.static_fts_encoder(fts)

    def select_best_from_virtual(self, node_states, scalars, batch):
        node_scalars = scalars[batch.edge_index[0] == batch.edge_index[1]]
        return self.select_best_virtual(node_states, node_scalars, batch.batch)

    def combined_edge_KV(self, node_states, edge_fts, scalars, batch):
        select_best = self.select_best_by_reciever(
            node_states[batch.edge_index[0]], scalars, batch.edge_index[1]
        )

        static_fts = self.compute_static_fts(scalars, batch)
        combined = self.combine_fts(
            torch.cat(
                [edge_fts, edge_fts[batch.batched_reverse_idx], static_fts], dim=1
            )
        )

        edge_K = self.edge_key(select_best)
        edge_V = self.edge_value(combined)

        return edge_K, edge_V

class ScalarUpdater(torch.nn.Module):
    def __init__(self, config: base_config.Config):
        super().__init__()
        h = config.h

        self.node_states_encoder = StatesEncoder(config.h, config.num_node_states)
        self.edge_states_encoder = StatesEncoder(config.h, config.num_edge_states)

        self.combine_fts = Linear(2 * h, h)

        self.keep_proj = Linear(h, 2)
        self.push_proj = Linear(h, 2)
        self.push_node_proj = Linear(h, 2)
        self.increment_proj = Linear(h, 2)

        self.scalars_only_as_input = config.generate_random_numbers
        self.temp = (
            config.processor_upper_t,
            config.processor_lower_t,
            config.num_iterations,
            config.temp_on_eval,
        )
        self.use_noise = config.use_noise

    def forward(
        self,
        node_states,
        edge_states,
        scalars,
        batch,
        training_step,
        processor_step,
        teacher_force,
    ):
        if self.scalars_only_as_input:
            return batch.scalars[:, processor_step], 0.0

        node_fts = self.node_states_encoder(node_states)
        edge_fts = self.edge_states_encoder(edge_states)

        fts = self.combine_fts(
            torch.cat(
                [edge_fts[batch.batched_reverse_idx], node_fts[batch.edge_index[0]]],
                dim=1,
            )
        )
        index = torch.repeat_interleave(torch.arange(fts.shape[0]).to(fts.device), 2)

        increment = self.compute_increment(fts, index, training_step)
        push = self.compute_push(fts, scalars.view(-1), batch, index, training_step)
        keep = self.compute_keep(fts, scalars.view(-1), index, training_step)

        new_scalars = torch.unsqueeze(increment + keep + push, -1)

        loss = (
            ((batch.scalars[:, processor_step] - new_scalars) ** 2).mean()
            if training_step != -1
            else 0.0
        )

        if teacher_force:
            new_scalars = batch.scalars[:, processor_step]

        return new_scalars, loss

    def compute_increment(self, fts, index, training_step):
        tau = temp_by_step(training_step, *self.temp)
        use_noise = self.use_noise and training_step != -1

        logits = self.increment_proj(fts).view(-1)
        increment = gumbel_softmax(logits, index=index, tau=tau, use_noise=use_noise)[
            ::2
        ]
        return 1.0 * increment

    def compute_push(self, fts, scalars, batch, index, training_step):
        tau = temp_by_step(training_step, *self.temp)
        use_noise = self.use_noise and training_step != -1

        push_without_node_logits = self.push_proj(fts).view(-1)
        push_without_node = gumbel_softmax(
            push_without_node_logits, index=index, tau=tau, use_noise=use_noise
        )[::2]

        push_with_node_logits = self.push_node_proj(fts).view(-1)
        push_with_node = gumbel_softmax(
            push_with_node_logits, index=index, tau=tau, use_noise=use_noise
        )[::2]

        node_scalars = scalars[batch.edge_index[0] == batch.edge_index[1]]
        scalars_without_node = scalars - node_scalars[batch.edge_index[1]]
        scalars_with_node = scalars_without_node + node_scalars[batch.edge_index[0]]

        edge_push_without_node = scatter(
            push_without_node * scalars_without_node, batch.edge_index[1], reduce="sum"
        )
        edge_push_with_node = scatter(
            push_with_node * scalars_with_node, batch.edge_index[1], reduce="sum"
        )

        accumulated_node = edge_push_without_node + edge_push_with_node
        edge_push = torch.zeros_like(scalars)
        edge_push[batch.edge_index[0] == batch.edge_index[1]] = accumulated_node
        return edge_push

    def compute_keep(self, fts, scalars, index, training_step):
        tau = temp_by_step(training_step, *self.temp)
        use_noise = self.use_noise and training_step != -1

        logits = self.keep_proj(fts).view(-1)
        keep = gumbel_softmax(logits, index=index, tau=tau, use_noise=use_noise)[::2]
        return scalars * keep


class StatesBottleneck(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config.h
        self.node_projections = ModuleList(
            [Linear(h, 1) for _ in range(config.num_node_states)]
        )
        self.edge_projections = ModuleList(
            [Linear(h, 1) for _ in range(config.num_edge_states)]
        )
        self.spec = SPEC[config.algorithm]

    def forward(
        self, node_fts, edge_fts, batch, training_step, processor_step, teacher_force
    ):
        states = []

        loss = 0.0


        #could probably be optimized for vector operations by having a group and projection dimension
        for group in range(2):# group=0: node, group=1: edge
            fts = node_fts if group == 0 else edge_fts
            stacked_fts = []

            projections = self.node_projections if group == 0 else self.edge_projections
            hints = (
                batch.node_fts[:, processor_step]
                if group == 0
                else batch.edge_fts[:, processor_step]
            )

            for idx, projection in enumerate(projections): # projections not a large matrix but n*[hx1] layers
                logits = projection(fts).squeeze()
                gt = hints[:, idx].double() #select the hint to be projected into

                # loss
                if training_step != -1:
                    if self.spec[group][idx] != MASK:
                        index = batch.batch if group == 0 else batch.edge_index[0]
                        weight = 1
                        if self.spec[group][idx] == EDGE_MASK_ONE:
                            index = batch.batch[batch.edge_index[0]]
                            num_nodes = (batch.batch == 0).sum()
                            weight = num_nodes
                        ce_loss = weight * node_pointer_loss(logits, gt, index)
                    else:
                        ce_loss = binary_cross_entropy_with_logits(logits, gt)

                    loss += ce_loss

                # postprocess
                if not teacher_force: #if not forced, we use the model's own predictions for the next step
                    if self.spec[group][idx] != MASK:
                        index = batch.batch if group == 0 else batch.edge_index[0]
                        if self.spec[group][idx] == EDGE_MASK_ONE:
                            index = batch.batch[batch.edge_index[0]]
                        pred = gumbel_softmax(
                            logits, index=index, tau=0.0, use_noise=False
                        )
                    else:
                        pred = 1.0 * (logits > 0.0)
                else:
                    pred = gt
                stacked_fts.append(torch.unsqueeze(pred, -1))
            states.append(torch.cat(stacked_fts, -1))

        return *states, loss


class DiscreteProcessor(torch.nn.Module):
    def __init__(self, config: base_config.Config):
        super().__init__()
        h = config.h
        self.message_passing = AttentionModule(config)

        self.node_ffn = Sequential(Linear(h, h), ReLU(), Linear(h, h), ReLU())
        self.edge_ffn = Sequential(Linear(2 * h, h), ReLU(), Linear(h, h), ReLU())

        self.states_bottleneck = StatesBottleneck(config)
        self.scalar_update = ScalarUpdater(config)

    def forward(
        self,
        node_states,
        edge_states,
        scalars,
        batch,
        training_step,
        processor_step,
        teacher_force,
    ):
        node_fts, edge_fts = self.message_passing( #also has the encoder inside it
            node_states, edge_states, scalars, batch, training_step
        )
        node_fts, edge_fts = self.ffn(node_fts, edge_fts, batch)

        node_states, edge_states, states_loss = self.states_bottleneck( #has part of the decoder inside it
            node_fts, edge_fts, batch, training_step, processor_step, teacher_force
        )
        out_scalars, scalars_loss = self.scalar_update( #has part of the decoder inside it
            node_states, #we use the discretized states for the update to be more consistent
            edge_states,
            scalars,
            batch,
            training_step,
            processor_step,
            teacher_force,
        )

        loss = scalars_loss + states_loss

        return node_states, edge_states, out_scalars, loss

    def ffn(self, node_fts, edge_fts, batch):
        node_fts = node_fts + self.node_ffn(node_fts)
        edge_fts_with_reversed = torch.cat(
            [edge_fts, edge_fts[batch.batched_reverse_idx]], dim=1
        )

        edge_fts = edge_fts + self.edge_ffn(edge_fts_with_reversed)
        return node_fts, edge_fts

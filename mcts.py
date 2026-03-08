import math
import torch
import numpy as np
import chess
from env import ACTION_SPACE_SIZE, encode_action, decode_action

class Node:
    def __init__(self, prior, to_play):
        self.visit_count = 0
        self.to_play = to_play # 1 for White, 0 for Black
        self.prior = prior
        self.value_sum = 0
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, net, args=None):
        self.net = net
        self.args = args or {
            'num_simulations': 100,
            'batch_size': 8,
            'virtual_loss': 3,
            'pb_c_base': 19652,
            'pb_c_init': 1.25,
            'root_dirichlet_alpha': 0.3, # chess typical
            'root_exploration_fraction': 0.25
        }
        self.nn_cache = {}

    @torch.no_grad()
    def evaluate_batch(self, state_tensors):
        device = next(self.net.parameters()).device
        state_tensor = torch.FloatTensor(state_tensors).to(device)
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            policy, value = self.net(state_tensor)
            
        policy = torch.softmax(policy, dim=1).to(torch.float32).cpu().numpy()
        value = value.to(torch.float32).cpu().numpy().flatten()
        return policy, value

    def search(self, env):
        # Optional: to prevent uncontrolled memory inflation between independent searches
        # but keep it if memory allows for massive multi-game speedups
        if len(self.nn_cache) > 100000:
            self.nn_cache.clear()
            
        root = Node(0, 1 if env.board.turn == chess.WHITE else 0)
        
        # Evaluate Root
        board_fen = env.board.fen()
        if board_fen in self.nn_cache:
            policy, _ = self.nn_cache[board_fen]
        else:
            policy_batch, value_batch = self.evaluate_batch(np.expand_dims(env.get_state(), 0))
            policy = policy_batch[0]
            self.nn_cache[board_fen] = (policy, value_batch[0])
        
        legal_actions = env.get_legal_actions()
        masked_policy = np.zeros_like(policy)
        for act in legal_actions:
            masked_policy[act] = policy[act]
            
        sum_policy = np.sum(masked_policy)
        if sum_policy > 0:
            masked_policy /= sum_policy
        else:
            for act in legal_actions:
                masked_policy[act] = 1.0 / len(legal_actions)
                
        # Expand root
        for act in legal_actions:
            root.children[act] = Node(masked_policy[act], 1 - root.to_play)
            
        if self.args.get('root_dirichlet_alpha', 0) > 0:
            noise = np.random.dirichlet([self.args.get('root_dirichlet_alpha')] * len(legal_actions))
            frac = self.args.get('root_exploration_fraction', 0)
            for i, act in enumerate(legal_actions):
                root.children[act].prior = root.children[act].prior * (1 - frac) + noise[i] * frac

        batch_size = self.args.get('batch_size', 8)
        num_batches = max(1, self.args['num_simulations'] // batch_size)
        virtual_loss = self.args.get('virtual_loss', 3)

        for _ in range(num_batches):
            paths = []
            leaf_states = []
            leaf_nodes = []
            leaf_legal_actions = []
            leaf_fens = []
            leaf_terminals = []
            
            for _ in range(batch_size):
                node = root
                search_path = [node]
                moves_pushed = 0
                
                # 1. Select
                while node.expanded():
                    action, node = self.select_child(node)
                    move = decode_action(action, env.board)
                    env.push(move)
                    moves_pushed += 1
                    search_path.append(node)
                    
                # Virtual loss for threaded exploration
                for n in search_path:
                    n.visit_count += virtual_loss
                    n.value_sum -= virtual_loss
                    
                paths.append(search_path)
                
                done = env.board.is_game_over(claim_draw=True)
                if not done:
                    # Try Transposition Table (MCTS Cache)
                    board_fen = env.board.fen()
                    if board_fen in self.nn_cache:
                        cached_policy, cached_val = self.nn_cache[board_fen]
                        legal_acts = env.get_legal_actions()
                        
                        masked_policy = np.zeros_like(cached_policy)
                        for act in legal_acts:
                            masked_policy[act] = cached_policy[act]
                            
                        sum_pol = np.sum(masked_policy)
                        if sum_pol > 0:
                            masked_policy /= sum_pol
                        else:
                            for act in legal_acts:
                                masked_policy[act] = 1.0 / len(legal_acts)
                                
                        for act in legal_acts:
                            node.children[act] = Node(masked_policy[act], 1 - node.to_play)
                            
                        leaf_terminals.append(cached_val)
                    else:
                        leaf_states.append(env.get_state())
                        leaf_nodes.append(node)
                        leaf_legal_actions.append(env.get_legal_actions())
                        leaf_fens.append(board_fen)
                        leaf_terminals.append(None)
                else:
                    result = env.board.result(claim_draw=True)
                    if result == '1-0':
                        val = 1.0 if node.to_play == 1 else -1.0
                    elif result == '0-1':
                        val = -1.0 if node.to_play == 1 else 1.0
                    else:
                        val = -0.8 # Punish heavily for draws to force aggressive winning attempts
                    leaf_terminals.append(val)
                    
                # Restore immediately
                for _ in range(moves_pushed):
                    env.pop()
                    
            # 2. Evaluate batch
            if leaf_states:
                batch_tensor = np.array(leaf_states)
                policies, values = self.evaluate_batch(batch_tensor)
                
                nn_idx = 0
                for i in range(batch_size):
                    if leaf_terminals[i] is None:
                        node = leaf_nodes[nn_idx]
                        policy = policies[nn_idx]
                        val = values[nn_idx]
                        legal_acts = leaf_legal_actions[nn_idx]
                        
                        masked_policy = np.zeros_like(policy)
                        for act in legal_acts:
                            masked_policy[act] = policy[act]
                            
                        sum_pol = np.sum(masked_policy)
                        if sum_pol > 0:
                            masked_policy /= sum_pol
                        else:
                            for act in legal_acts:
                                masked_policy[act] = 1.0 / len(legal_acts)
                                
                        for act in legal_acts:
                            node.children[act] = Node(masked_policy[act], 1 - node.to_play)
                            
                        # Save to Transposition Table Cache
                        fen = leaf_fens[nn_idx]
                        self.nn_cache[fen] = (policy, val)
                            
                        leaf_terminals[i] = val
                        nn_idx += 1
                        
            # 3. Backpropagate and remove virtual loss
            for i in range(batch_size):
                search_path = paths[i]
                val = leaf_terminals[i]
                
                for node in reversed(search_path):
                    node.visit_count += 1 - virtual_loss
                    node.value_sum += val + virtual_loss
                    val = -val

        visit_counts = np.zeros(ACTION_SPACE_SIZE)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count
            
        sum_visits = np.sum(visit_counts)
        if sum_visits > 0:
            action_probs = visit_counts / sum_visits
        else:
            action_probs = np.zeros(ACTION_SPACE_SIZE)
            if legal_actions:
                action_probs[np.random.choice(legal_actions)] = 1.0
                
        return action_probs

    def select_child(self, node):
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            score = self.ucb_score(node, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def ucb_score(self, parent, child):
        pb_c = math.log((parent.visit_count + self.args['pb_c_base'] + 1) / self.args['pb_c_base']) + self.args['pb_c_init']
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = -child.value()
        
        return value_score + prior_score

import math
import torch
import torch.multiprocessing as mp

from dataaccess.data_processor import VrpDataProcessor


class VrpModelSupervisor(object):
    def __init__(self, model, data_processor, args):
        self.model = model
        self.num_of_process = args.num_of_process
        self.global_step = 0
        self.dropout_rate = args.dropout_rate
        self.data_processor = VrpDataProcessor()
        self.batch_size = args.batch_size

    def load_pre_trained_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint)

    def train(self, batch_data):
        self.model.optimizer.zero_grad()
        avg_loss, avg_reward, dm_rec = self.model(batch_data)
        self.global_step += 1
        if type(avg_loss) != float:
            avg_loss.backward()
            self.model.train()
        return avg_loss.item(), avg_reward

    def batch_eval(self, eval_data, output_trace_flag, process_idx):
        cum_loss = 0
        cum_reward = 0
        data_size = len(eval_data)

        for batch_idx in range(0, data_size, self.batch_size):
            batch_data = self.data_processor.get_batch(eval_data, self.batch_size, batch_idx)
            cur_avg_loss, cur_avg_reward, dm_rec = self.model(batch_data, eval_flag=True)
            cum_loss += cur_avg_loss.item() * len(batch_data)
            cum_reward += cur_avg_reward * len(batch_data)
            # if output_trace_flag == 'complete':
            #     for cur_dm_rec in dm_rec:
            #         for i in range(len(cur_dm_rec)):
            #             print('step ' + str(i))
            #             dm = cur_dm_rec[i]
            #             print(dm.tot_dis[-1])
            #             for j in range(len(dm.vehicle_state)):
            #                 cur_pos, cur_capacity = dm.vehicle_state[j]
            #                 cur_node = dm.get_node(cur_pos)
            #                 print(cur_node.x, cur_node.y, cur_node.demand, cur_capacity, dm.tot_dis[j])
            #             print('')

            print('process start idx: %d batch idx: %d pred reward: %.4f' \
                  % (process_idx, batch_idx, cur_avg_reward))
        return cum_loss, cum_reward

    def eval(self, data, output_trace_flag, max_eval_size=None):
        data_size = len(data) if max_eval_size is None else min(len(data), max_eval_size)
        eval_data = data[:data_size]
        if self.num_of_process == 1:
            cum_loss, cum_reward = self.batch_eval(eval_data, output_trace_flag, 0)
        else:
            cum_loss = 0
            cum_reward = 0
            res = []
            try:
                mp.set_start_method('spawn')
            except RuntimeError:
                pass
            pool = mp.Pool(processes=self.num_of_process)
            sample_per_process = math.ceil(data_size/self.num_of_process)
            for st in range(0, data_size, sample_per_process):
                res += [pool.apply_async(self.batch_eval,
                                         (eval_data[st: st + sample_per_process], output_trace_flag, st))]
            for i in range(len(res)):
                cur_cum_loss, cur_cum_reward = res[i].get()
                cum_loss += cur_cum_loss
                cum_reward += cur_cum_reward

        avg_loss = cum_loss / data_size
        avg_reward = cum_reward / data_size
        print('average pred reward: %.4f' % avg_reward)
        return avg_loss, avg_reward

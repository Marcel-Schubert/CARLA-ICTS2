import os
import time
from matplotlib import pyplot as plt
import torch
from .model import A2C
from config import Config
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from benchmark.environment.ped_controller import l2_distance
import torch.nn as nn
from SAC.sac_discrete.sacd.utils import update_params, RunningMeanStats

from torch.utils.tensorboard import SummaryWriter

class A2CTrainer():
    def __init__(self, env, lr=0.0001, max_episodes=3000, eval_interval=1000, gamma=0.99,max_grad_norm=10, log_interval=10,
                  entropy_coef=0.005, value_coeff = 1.0,log_dir = "_out/a2c/", path=None,**kwargs):
        torch.manual_seed(100)
        self.network = A2C(hidden_dim=256, num_actions=3).cuda()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.max_episodes = max_episodes
        self.env = env
        self.current_episode = 0
        self.steps = 0
        self.eval_interval = eval_interval
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        #self.grad_norm = grad_norm
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm

        print(lr,gamma,entropy_coef,max_episodes,eval_interval)
        self.log_dir = log_dir + "_ec" + str(entropy_coef) + "_lr" + str(lr) + "_vc" + str(value_coeff)
        print(self.log_dir)
        self.model_dir = os.path.join(self.log_dir, 'model')
        self.summary_dir = os.path.join(self.log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        self.file = open(self.summary_dir + "eval_results.log", "w")
        if path is None:
            self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_return = RunningMeanStats(log_interval)
        self.log_interval = log_interval
        self.imitate = False
        if path is not None:
            self.network.load_state_dict(torch.load(path))



    def run(self):
        self.current_episode = 1
        best_cr = 100
        best_gr = 100
        while self.current_episode < self.max_episodes:
            self.train_episode()
            self.current_episode+=1
            if self.current_episode % self.eval_interval == 0 and self.steps > 400000:
                self.env.mode = "VALIDATION"
                cr, gr, avg_goal_vel = self.evaluate()
                if cr<best_cr and gr < best_gr and avg_goal_vel < 35: #was 40 # 30*1.1
                    best_cr = cr 
                    best_gr = gr
                    self.save_model()
                self.env.mode = "TRAINING"
            if self.current_episode % 250 == 0:
                print("Saved")
                self.save_model(str(self.current_episode))

    def save_model(self,post="best"):
        torch.save(self.network.state_dict(), self.model_dir+"/model_" + post+".pth")
        #self.network.save()

    def set_imitate(self):
        self.env.planner_agent.imitate = False
        self.imitate = False

    def unset_imitate(self):
        self.env.planner_agent.imitate = False
        self.imitate = False

    def train_episode(self):
        total_episode_reward = 0
        observation = self.env.reset()

        # Setup initial inputs for LSTM Cell
        cx = torch.zeros(1, 256).cuda().type(torch.cuda.FloatTensor)
        hx = torch.zeros(1, 256).cuda().type(torch.cuda.FloatTensor)

        # Setup placeholders for training value logs
        values = []
        log_probs = []
        rewards = []
        entropies = []

        reward = 0
        speed_action = 1
        velocity_x = 0
        velocity_y = 0
        nearmiss = False
        acccident = False
        episode_return = 0
        action_count = {0: 0, 1: 0, 2: 0}


        begin_pos = self.env.world.player.get_location()
        #start_speed = (velocity.x * velocity.x + velocity.y * velocity.y) ** 0.5
        #start_speed *= 3.6
        for step_num in range(Config.num_steps):
            # Forward pass of the RL Agent
            # if step_num > 0:
            #     plt.imsave('_out/{}.png'.format(step_num), observation)
            input_tensor = torch.from_numpy(observation).cuda().type(torch.cuda.FloatTensor)/255.0
            cat_tensor = torch.from_numpy(np.array([reward, velocity_x * 3.6, velocity_y * 3.6,
                                                    speed_action])).cuda().type(torch.cuda.FloatTensor)
            
            logit, value, (hx, cx) = self.network.forward(input_tensor, (hx, cx), cat_tensor)

            prob = F.softmax(logit, dim=-1)
            m = Categorical(prob)
            if self.imitate:
                speed_action = self.env.planner_agent.prev_planner_action
                action = torch.tensor(speed_action).cuda()
            else:
                action = m.sample()
                speed_action = action.item()
            action_count[speed_action] += 1

            observation, reward, done, info = self.env.step(speed_action)
            nearmiss_current = info['near miss']
            nearmiss = nearmiss_current or nearmiss
            acccident_current = info['accident']
            acccident = acccident_current or acccident
            total_episode_reward += reward

            # Logging value for loss calculation and backprop training
            log_prob = m.log_prob(action)
            entropy = -(F.log_softmax(logit, dim=-1) * prob).sum()
            velocity = info['velocity']
            velocity_x = velocity.x
            velocity_y = velocity.y
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)
            episode_return += reward

            if False:#self.plot_intention:
                #self.pc +=1
                print("Steps",self.steps)
                if self.steps % 20 == 0:
                    fig=plt.figure()
                    plt.imshow(observation)
                    fig.savefig("cp_debug/hylear_cp_%d.png"%self.steps,dpi=400)
                    print("Save",self.steps)
            if done or acccident:
                break
            self.steps +=1
        self.train_return.append(episode_return)
        end_pos = self.env.world.player.get_location()
        dist = l2_distance(begin_pos, end_pos)
        print(begin_pos,end_pos)
        print("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m".format(
            self.current_episode  + 1, info['scenario'], info['ped_speed'], info['ped_distance']))
        #file.write("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m\n".format(
        #    current_episode + 1, info['scenario'], info['ped_speed'], info['ped_distance']))
        print('Goal reached: {}, Accident: {}, Nearmiss: {}'.format(info['goal'], acccident, nearmiss))
        #file.write('Goal reached: {}, Accident: {}, Nearmiss: {}\n'.format(info['goal'], acccident, nearmiss))


        ##############################################################
        # Update weights of the model
        R = 0
        rewards.reverse()
        values.reverse()
        log_probs.reverse()
        entropies.reverse()
        returns = []
        for r in rewards:
            R = self.gamma * R + r
            returns.append(R)
        returns = torch.tensor(returns)
        eps = np.finfo(np.float32).eps.item()
        returns = (returns - returns.mean()) / (returns.std() + eps)
        returns = returns.cuda().type(torch.cuda.FloatTensor)

        policy_losses = []
        value_losses = []

        for log_prob, value, R in zip(log_probs, values, returns):
            advantage = R - value.item()
            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)
            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([[R]]).cuda()))
        loss = torch.stack(policy_losses).sum() + self.value_coeff * torch.stack(value_losses).sum() - \
               self.entropy_coef * torch.stack(entropies).sum()
        self.optimizer.zero_grad()
        loss.backward()
        #print(self.compute_policy_grad_norm())
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        #print(self.compute_policy_grad_norm())
        self.optimizer.step()
        print("Policy Loss: {:.4f}, Value Loss: {:.4f}, Entropy: {:.4f}, Reward: {:.4f}".format(
            torch.stack(policy_losses).sum().item(), torch.stack(value_losses).sum().item(),
            torch.stack(entropies).sum(), total_episode_reward))
        #file.write("Policy Loss: {:.4f}, Value Loss: {:.4f}, Reward: {:.4f}\n".format(
        #    torch.stack(policy_losses).sum().item(), torch.stack(value_losses).sum().item(), total_episode_reward))
        #if current_episode % Config.save_freq == 0:
        #    torch.save(self.network.state_dict(), "{}a2c_entropy_005_{}.pth".format(path, current_episode))
        print("Policy; ", action_count, "Distance: ", dist, "\n")

        if self.steps % 1 == 0:
            self.writer.add_scalar(
                'reward/train', self.train_return.get(), self.current_episode)
            self.writer.add_scalar(
                "loss/policy",torch.stack(policy_losses).sum().item(),
                self.current_episode
            )
            self.writer.add_scalar(
                "loss/value",torch.stack(value_losses).sum().item(),
                self.current_episode
            )
            self.writer.add_scalar(
                "stats/entropy",torch.stack(entropies).sum(),
                self.current_episode
            )
            self.writer.add_scalar(
                'reward/train', total_episode_reward,
                self.current_episode)
            self.writer.add_scalar(
                'stats/dist', dist,
                self.current_episode)
            
            self.writer.add_scalar(
                'loss/grad', self.compute_policy_grad_norm(),
                self.current_episode
            )



        

    def update(self):
        pass

    def compute_policy_grad_norm(self):
        total_norm = 0
        with torch.no_grad():
            for p in self.network.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
        return total_norm
    
    def evaluate(self, mode="VALIDATION"):

        self.env.mode = mode
        if mode=="VALIDATION":
            num_episodes = len(self.env.val_episodes)
        else:
            num_episodes = len(self.env.test_episodes)
        #num_episodes = 20

        print('-' * 60)
        #self.env.reset_eval_iterator()
        print(self.env.mode, num_episodes)
        data_log = {}
        num_accidents = 0
        num_nearmisses = 0
        num_goals = 0
        velocity_goal = 0
        #current_episode = 0
        time_to_goal_list = []
        exec_time_list = []
        for current_episode in range(1,num_episodes+1):
            # Get the scenario id, parameters and instantiate the world
            total_episode_reward = 0
            observation = self.env.reset()

            # Setup initial inputs for LSTM Cell
            cx = torch.zeros(1, 256).cuda().type(torch.cuda.FloatTensor)
            hx = torch.zeros(1, 256).cuda().type(torch.cuda.FloatTensor)

            # Setup placeholders for training value logs
            values = []
            log_probs = []
            rewards = []
            entropies = []

            reward = 0
            speed_action = 1
            velocity_x = 0
            velocity_y = 0
            nearmiss = False
            acccident = False

            action_count = {0: 0, 1: 0, 2: 0}
            

            nearmiss = False
            begin_pos = self.env.world.player.get_location()

            for step_num in range(Config.num_steps):
                if Config.display:
                    self.env.render()
                    # Forward pass of the RL Agent
                    # if step_num > 0:
                    #     plt.imsave('_out/{}.png'.format(step_num), observation)
                input_tensor = torch.from_numpy(observation).cuda().type(torch.cuda.FloatTensor)/255.0
                cat_tensor = torch.from_numpy(np.array([reward, velocity_x * 3.6, velocity_y * 3.6,
                                                            speed_action])).cuda().type(torch.cuda.FloatTensor)
                start_time = time.time()
                with torch.no_grad():
                    logit, value, (hx, cx) = self.network.forward(input_tensor, (hx, cx), cat_tensor)

                prob = F.softmax(logit, dim=-1)
                m = Categorical(prob)
                #action = m.sample()
                #speed_action = action.item()
                speed_action = torch.argmax(logit, dim=1)[0].item()
                action = torch.tensor(speed_action).cuda()
                action_count[speed_action] += 1

                observation, reward, done, info = self.env.step(speed_action)
                time_taken = time.time() - start_time
                exec_time_list.append(time_taken*1000)

                nearmiss_current = info['near miss']
                nearmiss = nearmiss_current or nearmiss
                acccident_current = info['accident']
                acccident = acccident_current or acccident
                total_episode_reward += reward

                # Logging value for loss calculation and backprop training
                log_prob = m.log_prob(action)
                entropy = -(F.log_softmax(logit, dim=-1) * prob).sum()
                velocity = info['velocity']
                velocity_x = velocity.x
                velocity_y = velocity.y
                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy)
                nearmiss = nearmiss or info["near miss"]
                if done or acccident:
                    break
                
            R = 0
            rewards.reverse()
            values.reverse()
            log_probs.reverse()
            entropies.reverse()
            returns = []
            for r in rewards:
                R = self.gamma * R + r
                returns.append(R)
            returns = torch.tensor(returns)
            eps = np.finfo(np.float32).eps.item()
            returns = (returns - returns.mean()) / (returns.std() + eps)
            returns = returns.cuda().type(torch.cuda.FloatTensor)

            policy_losses = []
            value_losses = []

            for log_prob, value, R in zip(log_probs, values, returns):
                advantage = R - value.item()
                # calculate actor (policy) loss
                policy_losses.append(-log_prob * advantage)
                # calculate critic (value) loss using L1 smooth loss
                value_losses.append(F.smooth_l1_loss(value, torch.tensor([[R]]).cuda()))
            loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum() - \
                    Config.a2c_entropy_coef * torch.stack(entropies).sum()
            speed = (velocity.x * velocity.x + velocity.y * velocity.y) ** 0.5
            speed *= 3.6
            time_to_goal = step_num * Config.simulation_step
            time_to_goal_list.append(time_to_goal)
            num_accidents += 1 if info["accident"] else 0
            num_nearmisses += 1 if nearmiss and not info["accident"] else 0
            num_goals += 1 if info['goal'] else 0
            velocity_goal +=  speed if info['goal'] else 0
            end_pos = self.env.world.player.get_location()
            dist = l2_distance(begin_pos, end_pos)


            print(begin_pos,end_pos)
            print("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m".format(
            current_episode + 1, info['scenario'], info['ped_speed'], info['ped_distance']))
            print('Goal reached: {}, Accident: {}, Nearmiss: {}'.format(info['goal'], acccident, nearmiss)) 
            print("Policy Loss: {:.4f}, Value Loss: {:.4f}, Entropy: {:.4f}, Reward: {:.4f}".format(
            torch.stack(policy_losses).sum().item(), torch.stack(value_losses).sum().item(),
            torch.stack(entropies).sum(), total_episode_reward))  
            print("Reward: {:.4f}".format( total_episode_reward))
            print("Policy; ", action_count, "Distance: ", dist)
            print("Crash Rate: {:.2f}, Nearmiss Rate: {:.2f}, Goal Rate: {:.2f}".format(
                num_accidents/current_episode, num_nearmisses/current_episode, num_goals/current_episode))
            print("Velocity Goal: {:.4f}, Exec time: {:.4f},TTG: {:.4f}".format(velocity_goal/current_episode, np.mean(exec_time_list), np.mean(time_to_goal_list)))
            print(current_episode)

        print('-' * 60)
        print(f"{'avg. crash rate:':20}{num_accidents/current_episode:.2f}%")
        print(f"{'avg. nearmiss rate:':20}{num_nearmisses/current_episode:.2f}%")
        print(f"{'avg. goal rate:':20}{num_goals/current_episode:.2f}%")
        print(f"{'avg. time to goal:':20}{np.mean(time_to_goal_list):.2f}s")
        print(f"{'avg. exec time:':20}{np.mean(exec_time_list):.2f}ms")
        print('-' * 60)

        # print(round(num_accidents/current_episode,2), " % ",
        #   round(num_nearmisses/current_episode,2), " % ",
        #   round(num_goals/current_episode,2)," % ",
        #   round(np.mean(time_to_goal_list),2), " % ",
        #   round(np.mean(exec_time_list),2), " % ",
        #   )
        # print('-' * 60)
        self.env.mode = "TRAINING"
        self.env.reset_iterator()

        return num_accidents/current_episode, num_goals/current_episode, velocity_goal/current_episode
import numpy as np
import utils
import random
import copy

class Agent:

	def __init__(self, actions, Ne, C, gamma):
		self.actions = actions
		self.Ne = Ne  # used in exploration function
		self.C = C
		self.gamma = gamma

		# Create the Q and N Table to work with
		self.Q = utils.create_q_table()
		self.N = utils.create_q_table()
			
		self.reset()

	def train(self):
		self._train = True

	def eval(self):
		self._train = False

	# At the end of training save the trained model
	def save_model(self, model_path):
		utils.save(model_path, self.Q)

	# Load the trained model for evaluation
	def load_model(self, model_path):
		self.Q = utils.load(model_path)

	def reset(self):
		self.points = 0
		self.s = None
		self.a = None

	def discretize(self, state):
		# discretize
		snake_head_x, snake_head_y, snake_body, food_x, food_y = state
		# State: A tuple (adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right).
		# snake_head_x = snake_head_x / utils.GRID_SIZE
		# snake_head_y = snake_head_y / utils.GRID_SIZE
		# food_x = food_x / utils.GRID_SIZE
		# food_y = food_y / utils.GRID_SIZE

		if snake_head_x < 1 * utils.GRID_SIZE or snake_head_x > 13 * utils.GRID_SIZE \
			or snake_head_y < 1 * utils.GRID_SIZE or snake_head_y > 13 * utils.GRID_SIZE:
			adjoining_wall_x = 0
			adjoining_wall_y = 0
		else:
			adjoining_wall_x = 0
			if snake_head_x == 1 * utils.GRID_SIZE:
				adjoining_wall_x = 1
			# if snake_head_x == 13 * utils.GRID_SIZE:
			# 	adjoining_wall_x = 2
			if snake_head_x == 12 * utils.GRID_SIZE:
				adjoining_wall_x = 2
			# print(adjoining_wall_x)

			adjoining_wall_y = 0
			if snake_head_y == 1 * utils.GRID_SIZE:
				adjoining_wall_y = 1
			# if snake_head_y == 13 * utils.GRID_SIZE:
			# 	adjoining_wall_y = 2
			if snake_head_y == 12 * utils.GRID_SIZE:
				adjoining_wall_y = 2
			# print(adjoining_wall_y)
		# Food dir
		if (snake_head_x - food_x) > 0:
			food_dir_x = 1
		elif (snake_head_x - food_x) == 0:
			food_dir_x = 0
		else:
			food_dir_x = 2

		if (snake_head_y - food_y) > 0:
			food_dir_y = 1
		elif (snake_head_y - food_y) == 0:
			food_dir_y = 0
		else:
			food_dir_y = 2

		# snake body
		adjoining_body_top = 0
		adjoining_body_bottom = 0
		adjoining_body_left = 0
		adjoining_body_right = 0

		for (cell_x, cell_y) in snake_body:
			if cell_x == snake_head_x + utils.GRID_SIZE and cell_y == snake_head_y:
				adjoining_body_right = 1

			if cell_x == snake_head_x - utils.GRID_SIZE and cell_y == snake_head_y:
				adjoining_body_left = 1

			if cell_y == snake_head_y + utils.GRID_SIZE and cell_x == snake_head_x:
				adjoining_body_bottom = 1

			if cell_y == snake_head_y - utils.GRID_SIZE and cell_x == snake_head_x:
				adjoining_body_top = 1

		state_discretized = [adjoining_wall_x, adjoining_wall_y, food_dir_x, food_dir_y,
						 adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right]
			# print(state_discretized)
		return state_discretized

	def maxQ(self, state):
		# print('hi')
		val = -1e10
		state = self.discretize(state)
		for a in [3, 2, 1, 0]:
			sa = state[:]
			sa.append(a)
			indices = tuple(sa)
			# print(sa)
			if self.Q[indices] > val:
				val = self.Q[indices]

		return val

	def act(self, state, points, dead):
		'''
		:param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
		:param points: float, the current points from environment
		:param dead: boolean, if the snake is dead
		:return: the index of action. 0,1,2,3 indicates up,down,left,right separately

		TODO: write your function here.
		Return the index of action the snake needs to take, according to the state and points known from environment.
		Tips: you need to discretize the state to the state space defined on the webpage first.
		(Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

		'''
		snake_head_x, snake_head_y, snake_body, food_x, food_y = state


		s_prime = self.discretize(state)

		if self._train and self.s is not None and self.a is not None:
			"""During training, your agent needs to update your Q-table first
			(this step is skipped when the initial state and action are None),
			"""

			# alpha = self.C / (self.C + self.N[self.s, self.a])
			sa = self.discretize(self.s)[:]
			sa.append(self.a)
			indices = tuple(sa)

			alpha = self.C / (self.C + self.N[indices])

			# reward
			if points - self.points > 0:
				r = 1
			elif dead:
				r = -1
			else:
				r = -0.1

			# Next action

			self.Q[indices] += alpha * \
				(r + self.gamma * self.maxQ(state) - self.Q[indices])

		if dead:
			""" If the game is over, that is when
			the dead varaible becomes true, you only need to update your Q table
			and reset the game."""
			# print('dead')
			self.reset()
			return 0

		"""get the next action using the above exploration policy, and then 
			update N-table with that action. During testing, your agent only needs to give the 
			best action using Q-table."""

		def f(u, n):
			if n < self.Ne and self._train:
				return 1
			else:
				return u

		a = None
		a_val = 1e-10
		for a_prime in [3,2,1,0]:
			# {up, down, left, right}
			sli = s_prime[:]
			sli.append(a_prime)
			indices = tuple(sli)

			if a is None:
				a = a_prime
				# print(self.N[indices])
				a_val = f(self.Q[indices], self.N[indices])
				continue

			val = f(self.Q[indices], self.N[indices])

			if val > a_val:
				a = a_prime
				a_val = val

		sli = s_prime[:]
		sli.append(a)
		indices = tuple(sli)
		self.N[indices] += 1
		
		self.s = copy.deepcopy(state)
		self.a = copy.deepcopy(a)
		self.points = points
		return self.actions[a]
		# return 0

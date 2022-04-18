'''
BlindBot MazeAgent meant to employ Propositional Logic,
Search, Planning, and Active Learning to navigate the
Maze Pitfall problem
Created by: Chase, Andrew, and Tommy
'''

import time
import random
from pathfinder import *
from maze_problem import *
from queue import Queue
from maze_clause import *
from maze_knowledge_base import *
from constants import *
from numpy import *
import copy

# [!] TODO: import your Problem 1 when ready here!

class MazeAgent:

    ##################################################################
    # Constructor
    ##################################################################

    def __init__ (self, env):
        """
        Initializes the maze agent, along with key information that he will store
        in order to make decisions as he traverses through the maze
        :env: The maze environment that our agent is navigating through
        :loc: The current x-y tuple location of our agent in the maze
        :goal: The location of the goal within our given environment
        :walls: The location of the boundary walls within the maze
        :kb: The knowledge base that our agent uses to make his decisions,
        which has been supplemented with the knowledge that the goal is not a pit
        :rows: The number of rows within the maze
        :cols: The number of columns within the maze
        :plan: A queue of actions that the agent is designated to take
        """
        self.env  = env
        self.loc  = env.get_player_loc()
        self.goal = env.get_goal_loc()
        self.walls = copy.deepcopy(env._walls)
        self.kb = MazeKnowledgeBase()
        self.kb.tell(MazeClause([(("P", env.get_goal_loc()),False)]))
        self.rows = copy.deepcopy(env._rows)
        self.cols = copy.deepcopy(env._cols)
        self.maze = env.get_agent_maze()
        self.safeSpace = []
        self.plan = Queue()

    ##################################################################
    # Methods
    ##################################################################

    def clearPath(self,uk_space):
        """
        This method allows our agent to determine if there is a clear path to a
        given location within our maze. If there is a clear path that is not blocked
        by any potential pit locations, our method will return True. Otherwise, we
        return False
        :uk_space: A given "unknown" space that we are trying to find a path to
        """
        tempPlan = self.make_plan(uk_space)
        if self.test_plan(uk_space,tempPlan):
            return True
        else:
            return False
    def last_loc(self, move):
        """
        A method that simply returns the last location of our agent before his
        previous action. Uses his last action to backtrack and determine the
        previous location
        :move: The previous move that our agent took in the maze
        """
        lLoc = self.loc
        if move == "U":
            lLoc = (lLoc[0],lLoc[1] - 1)
        elif move == "D":
            lLoc = (lLoc[0],lLoc[1] + 1)
        elif move == "R":
            lLoc = (lLoc[0] - 1,lLoc[1])
        elif move == "L":
            lLoc = (lLoc[0] + 1,lLoc[1])
        return lLoc
    def make_plan(self,goal):
        """
        A method that allows our agent to make a plan of action towards a "goal"
        space (not necessarily the goal state itself) by using an implementation
        of the Pathfinder class.
        :goal: The space our agent is attempting to make a plan to reach
        """
        tempPlan = Queue()
        tempPF = pathfind(MazeProblem(self.maze),self.loc,goal)
        tempPFarr = tempPF[1]
        for i in tempPFarr:
            tempPlan.put(i)
        return tempPlan
    def test_plan(self,goal,plan):
        """
        This method allows our agent to test any plan that he has created. We use
        a tempPlan in this method in order for our agent to avoid altering the
        current plan he may be following and still be able to test a new one. If
        a plan is safe to carry out, we return True. Otherwise, we return False.
        :goal: The space our agent will reach by the end of carrying out the plan
        :plan: The plan (or sequence of actions) our agent is testing to be safe
        """
        tempLoc = copy.deepcopy(self.loc)
        tempPlan = Queue()
        while tempLoc != goal:
            nm = plan.get()
            tempLoc = self.loc_after_move(tempLoc,nm)
            if self.kb.ask(MazeClause([(("P",tempLoc),False)])):
                tempPlan.put(nm)
            else:
                return False
        return True
    def loc_after_move(self, loc, move):
        """
        A simple method that we use in order to determine the location our agent
        will be at after making a move in the maze.
        :loc: The current x-y tuple location of the agent in his environment
        :move: The action our agent is taking to change his location
        """
        lLoc = loc
        if move == "U":
            lLoc = (lLoc[0],lLoc[1] - 1)
        elif move == "D":
            lLoc = (lLoc[0],lLoc[1] + 1)
        elif move == "R":
            lLoc = (lLoc[0] + 1,lLoc[1])
        elif move == "L":
            lLoc = (lLoc[0] - 1,lLoc[1])
        return lLoc
    def createClause(self,tile):
        """
        An important method that allows our agent to create a MazeClause that can
        be inserted into our KB. This method is called to deal with tiles that hold
        either a 1 or 2 integer value
        :tile: The integer tile space being used to create a MazeClause
        """
        x = self.loc[0]
        y = self.loc[1]
        mv = int(tile)
        locs = [(x+mv,y),(x-mv,y),(x,y+mv),(x,y-mv)]
        locsLeft = []
        for i in range (0,len(locs)):
            if locs[i][0] < 1 or locs[i][0] > self.env._cols - 2:
                locsLeft.insert(0,i)
            elif locs[i][1] < 1 or locs[i][1] > self.env._rows - 2:
                locsLeft.insert(0,i)
            elif self.kb.ask(MazeClause([(("P",locs[i]),False)])):
                locsLeft.insert(0,i)
        for i in locsLeft:
            del locs[i]
        arr = []
        for i in locs:
            arr.append((("P",i),True))
        return MazeClause(arr)
    def perceive(self,perception):
        """
        This method allows our agent to better understand his environment as he
        explores the maze. Takes in a "perception" from our agent and uses this
        to determine the safe spaces around our agent that should be explored
        :perception: The agent's location and current tile being stood on
        """
        x = self.loc[0]
        y = self.loc[1]
        self.kb.tell(MazeClause([((perception.get('tile'),self.loc),True)]))
        if perception.get('tile') != "P":
            self.kb.tell(MazeClause([(("P",self.loc),False)]))
        if perception.get('tile') == "2":
            self.kb.tell(self.createClause(perception.get('tile')))
            self.kb.tell(MazeClause([(("P", (x+1,y)),False)]))
            self.kb.tell(MazeClause([(("P", (x-1,y)),False)]))
            self.kb.tell(MazeClause([(("P", (x,y-1)),False)]))
            self.kb.tell(MazeClause([(("P", (x,y+1)),False)]))
            self.addToSafeSpace(x,y,1)
        elif perception.get('tile') == "1":
            self.kb.tell(self.createClause(perception.get('tile')))
        elif perception.get('tile') == ".":
            self.kb.tell(MazeClause([(("P", (x+2,y)),False)]))
            self.kb.tell(MazeClause([(("P", (x-2,y)),False)]))
            self.kb.tell(MazeClause([(("P", (x,y-2)),False)]))
            self.kb.tell(MazeClause([(("P", (x,y+2)),False)]))
            self.kb.tell(MazeClause([(("P", (x+1,y)),False)]))
            self.kb.tell(MazeClause([(("P", (x-1,y)),False)]))
            self.kb.tell(MazeClause([(("P", (x,y-1)),False)]))
            self.kb.tell(MazeClause([(("P", (x,y+1)),False)]))
            self.addToSafeSpace(x,y,2)
    def addToSafeSpace(self,x,y,range):
        """
        A method that we created in order to allow our agent to have a better sense
        of his environment. Once our agent has determined the spaces around him that
        are safe, he adds them to an array of spaces to be explored.
        :x: The current x coordinate of our agent
        :y: The current y coordinate of our agent
        :range: The range of safe tiles around our agent that do not contain a pit
        """
        while range > 0:
            if not (x+range<1 or x+range>self.env._cols-2) and not (y < 1 or y > self.env._rows - 2) and not((x+range,y) in self.safeSpace) and self.clearPath((x+range,y)):
                if self.env._ag_maze[y][x+range] == "?":
                    self.safeSpace.append((x+range,y))
            if not (x-range<1 or x-range>self.env._cols-2) and not (y < 1 or y > self.env._rows - 2) and not((x-range,y) in self.safeSpace) and self.clearPath((x-range,y)):
                if self.env._ag_maze[y][x-range] == "?":
                    self.safeSpace.append((x-range,y))
            if not (x<1 or x>self.env._cols-2) and not (y+range< 1 or y+range>self.env._rows - 2) and not((x,y+range) in self.safeSpace) and self.clearPath((x,y+range)):
                if self.env._ag_maze[y+range][x] == "?":
                    self.safeSpace.append((x,y+range))
            if not (x< 1 or x> self.env._cols-2) and not (y-range< 1 or y-range> self.env._rows - 2) and not((x,y-range) in self.safeSpace) and self.clearPath((x,y-range)):
                if self.env._ag_maze[y-range][x] == "?":
                    self.safeSpace.append((x,y-range))
            range=range-1
        return
    def addKnowns(self):
        """
        A method that updates our agent's environment with information from the
        KB. Because of the time complexity of continually doing this we decided
        to turn this into a helper method so computation time is lowered
        """
        for i in range(1,self.env._rows - 1):
            for j in range(1,self.env._cols - 1):
                if self.kb.ask(MazeClause([(("P",(j,i)),True)])):
                    self.env._ag_maze[i][j] = "P"
                    if self.kb.ask(MazeClause([(("P",(j+1,i)),False)])) and (j+1,i) not in self.walls and self.env._ag_maze[i][j+1] == "?":
                        self.env._ag_maze[i][j+1] = "1"
                        if (j+1, i) in self.safeSpace:
                            self.safeSpace.remove((j+1,i))
                    if self.kb.ask(MazeClause([(("P",(j-1,i)),False)])) and (j-1,i) not in self.walls and self.env._ag_maze[i][j-1] == "?":
                        self.env._ag_maze[i][j-1] = "1"
                        if (j-1, i) in self.safeSpace:
                            self.safeSpace.remove((j-1,i))
                    if self.kb.ask(MazeClause([(("P",(j,i+1)),False)])) and (j,i+1) not in self.walls and self.env._ag_maze[i+1][j] == "?":
                        self.env._ag_maze[i+1][j] = "1"
                        if (j,i+1) in self.safeSpace:
                            self.safeSpace.remove((j,i+1))
                    if self.kb.ask(MazeClause([(("P",(j,i-1)),False)])) and (j,i-1) not in self.walls and self.env._ag_maze[i-1][j] == "?":
                        self.env._ag_maze[i-1][j] = "1"
                        if (j,i-1) in self.safeSpace:
                            self.safeSpace.remove((j,i-1))
    def getLowestSS(self):
        """
        This method allows our agent to make informed explorations by finding the
        closest safe spaces to its current location. Implements the heuristic from
        Pathfinder rather than the pathfind method itself to lower computation time
        """
        lowestSS = (-1,-1)
        tempLow = 999999999999999999999999999999
        for i in self.safeSpace:
            tempPF = heuristic(self.loc,i)
            if tempPF < tempLow:
                lowestSS = i
                tempLow = tempPF
            elif tempPF == tempLow:
                if heuristic(self.goal,i) < heuristic(self.goal,lowestSS):
                    lowestSS = i
                    tempLow = tempPF
        self.safeSpace.remove(lowestSS)
        return lowestSS
    def think(self, perception):
        """
        think is parameterized by the agent's perception of the tile type
        on which it is now standing, and is called during the environment's
        action loop. This method is the chief workhorse of your MazeAgent
        such that it must then generate a plan of action from its current
        knowledge about the environment.
        The bot updates his location, and calls perceive() to add any new
        information about the immediate surrounding area. Then if the plan is
        empty, then it updates our map with any new pits that it learned about
        using addKnowns(). No matter what it checks if there is a clearPath()
        to the goal from where it is at, if so he makes that the plan. If the
        plan is not empty, then it return because we know what the next move is.
        If plan is empty and there is no clear path to the goal, then we checks
        to find the best safe sppace to explore using getLowestSS(). Our last
        option is if there are no more safe spaces, then he starts walking to
        the goal until he finds out more information and can get back to making
        educated decisions on where to go next.
        :perception: A dictionary providing the agent's current location
        and current tile type being stood upon, of the format:
          {"loc": (x, y), "tile": tile_type}
        """

        # Agent simply moves randomly at the moment...
        # Do something that thinks about the perception!
        self.loc = perception.get('loc')
        self.perceive(perception)
        if self.plan.empty():
            self.addKnowns()
        if self.loc in self.safeSpace:
            self.safeSpace.remove(self.loc)
        if self.clearPath(self.goal):
            self.plan = Queue()
            self.plan = self.make_plan(self.goal)
        elif not self.plan.empty():
            return
        elif len(self.safeSpace) != 0:
            temp = self.getLowestSS()
            self.plan = self.make_plan(temp)
        else:
            self.plan.put(self.make_plan(self.goal).get())

    def get_next_move(self):
        """
        Returns the next move in the plan, if there is one, otherwise None
        [!] You should NOT need to modify this method -- contact Dr. Forney
            if you're thinking about it
        """
        return None if self.plan.empty() else self.plan.get()

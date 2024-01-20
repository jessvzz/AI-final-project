#DISPLAYABLE + SearchProblem + cspProblem
import random
import matplotlib.pyplot as plt


class Displayable(object):
	"""Class that uses 'display'.
	The amount of detail is controlled by max_display_level
	"""
	max_display_level = 10 # can be overridden in subclasses

	def display(self,level,*args,**nargs):
		"""print the arguments if level is less than or equal to the
		current max_display_level.
		level is an integer.
		the other arguments are whatever arguments print can take.
		"""
		if level <= self.max_display_level:
			print(*args, **nargs) ##if error you are using Python2 not Python3

def visualize(func):
	"""A decorator for algorithms that do interactive visualization. Ignored here.
	"""
	return func


class Search_problem(object):
    """A search problem consists of:
    * a start node
    * a neighbors function that gives the neighbors of a node
    * a specification of a goal
    * a (optional) heuristic function.
    The methods must be overridden to define a search problem."""

    def start_node(self):
        """returns start node"""
        raise NotImplementedError("start_node")   # abstract method

    def is_goal(self,node):
        """is True if node is a goal"""
        raise NotImplementedError("is_goal")   # abstract method

    def neighbors(self,node):
        """returns a list of the arcs for the neighbors of node"""
        raise NotImplementedError("neighbors")   # abstract method

    def heuristic(self,n):
        """Gives the heuristic value of node n.
        Returns 0 if not overridden."""
        return 0

class Arc(object):
    """An arc has a from_node and a to_node node and a (non-negative) cost"""
    def __init__(self, from_node, to_node, cost=1, action=None):
        assert cost >= 0, ("Cost cannot be negative for"+
                           str(from_node)+"->"+str(to_node)+", cost: "+str(cost))
        self.from_node = from_node
        self.to_node = to_node
        self.action = action
        self.cost=cost

    def __repr__(self):
        """string representation of an arc"""
        if self.action:
            return str(self.from_node)+" --"+str(self.action)+"--> "+str(self.to_node)
        else:
            return str(self.from_node)+" --> "+str(self.to_node)

class Search_problem_from_explicit_graph(Search_problem):
    """A search problem consists of:
    * a list or set of nodes
    * a list or set of arcs
    * a start node
    * a list or set of goal nodes
    * a dictionary that maps each node into its heuristic value.
    * a dictionary that maps each node into its (x,y) position
    """

    def __init__(self, nodes, arcs, start=None, goals=set(), hmap={}, positions={}):
        self.neighs = {}
        self.nodes = nodes
        for node in nodes:
            self.neighs[node]=[]
        self.arcs = arcs
        for arc in arcs:
            self.neighs[arc.from_node].append(arc)
        self.start = start
        self.goals = goals
        self.hmap = hmap
        self.positions = positions

    def start_node(self):
        """returns start node"""
        return self.start

    def is_goal(self,node):
        """is True if node is a goal"""
        return node in self.goals

    def neighbors(self,node):
        """returns the neighbors of node"""
        return self.neighs[node]

    def heuristic(self,node):
        """Gives the heuristic value of node n.
        Returns 0 if not overridden in the hmap."""
        if node in self.hmap:
            return self.hmap[node]
        else:
            return 0

    def __repr__(self):
        """returns a string representation of the search problem"""
        res=""
        for arc in self.arcs:
            res += str(arc)+".  "
        return res

    def neighbor_nodes(self,node):
        """returns an iterator over the neighbors of node"""
        return (path.to_node for path in self.neighs[node])

class Path(object):
    """A path is either a node or a path followed by an arc"""

    def __init__(self,initial,arc=None):
        """initial is either a node (in which case arc is None) or
        a path (in which case arc is an object of type Arc)"""
        self.initial = initial
        self.arc=arc
        if arc is None:
            self.cost=0
        else:
            self.cost = initial.cost+arc.cost

    def end(self):
        """returns the node at the end of the path"""
        if self.arc is None:
            return self.initial
        else:
            return self.arc.to_node

    def nodes(self):
        """enumerates the nodes for the path.
        This starts at the end and enumerates nodes in the path backwards."""
        current = self
        while current.arc is not None:
            yield current.arc.to_node
            current = current.initial
        yield current.initial

    def initial_nodes(self):
        """enumerates the nodes for the path before the end node.
        This starts at the end and enumerates nodes in the path backwards."""
        if self.arc is not None:
            yield from self.initial.nodes()

    def __repr__(self):
        """returns a string representation of a path"""
        if self.arc is None:
            return str(self.initial)
        elif self.arc.action:
            return (str(self.initial)+"\n   --"+str(self.arc.action)
                    +"--> "+str(self.arc.to_node))
        else:
            return str(self.initial)+" --> "+str(self.arc.to_node)

class Variable(object):
    """A random variable.
    name (string) - name of the variable
    domain (list) - a list of the values for the variable.
    Variables are ordered according to their name.
    """

    def __init__(self, name, domain, position=None):
        """Variable
        name a string
        domain a list of printable values
        position of form (x,y)
        """
        self.name = name   # string
        self.domain = domain # list of values
        self.position = position if position else (random.random(), random.random())
        self.size = len(domain)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name  # f"Variable({self.name})"

class Constraint(object):
    """A Constraint consists of
    * scope: a tuple of variables
    * condition: a Boolean function that can applied to a tuple of values for variables in scope
    * string: a string for printing the constraints. All of the strings must be unique.
    for the variables
    """
    def __init__(self, scope, condition, string=None, position=None):
        self.scope = scope
        self.condition = condition
        if string is None:
            self.string = self.condition.__name__ + str(self.scope)
        else:
            self.string = string
        self.position = position

    def __repr__(self):
        return self.string

    def can_evaluate(self, assignment):
        """
        assignment is a variable:value dictionary
        returns True if the constraint can be evaluated given assignment
        """
        return all(v in assignment for v in self.scope)

    def holds(self,assignment):
        """returns the value of Constraint con evaluated in assignment.

        precondition: all variables are assigned in assignment, ie self.can_evaluate(assignment) is true
        """
        return self.condition(*tuple(assignment[v] for v in self.scope))

class CSP(object):
    """A CSP consists of
    * a title (a string)
    * variables, a set of variables
    * constraints, a list of constraints
    * var_to_const, a variable to set of constraints dictionary
    """
    def __init__(self, title, variables, constraints):
        """title is a string
        variables is set of variables
        constraints is a list of constraints
        """
        self.title = title
        self.variables = variables
        self.constraints = constraints
        self.var_to_const = {var:set() for var in self.variables}
        for con in constraints:
            for var in con.scope:
                self.var_to_const[var].add(con)

    def __str__(self):
        """string representation of CSP"""
        return str(self.title)

    def __repr__(self):
        """more detailed string representation of CSP"""
        return f"CSP({self.title}, {self.variables}, {([str(c) for c in self.constraints])})"

    def consistent(self,assignment):
        """assignment is a variable:value dictionary
        returns True if all of the constraints that can be evaluated
                        evaluate to True given assignment.
        """
        return all(con.holds(assignment)
                    for con in self.constraints
                    if con.can_evaluate(assignment))

    def show(self):
        plt.ion()   # interactive
        ax = plt.figure().gca()
        ax.set_axis_off()
        plt.title(self.title)
        var_bbox = dict(boxstyle="round4,pad=1.0,rounding_size=0.5")
        con_bbox = dict(boxstyle="square,pad=1.0",color="green")
        for var in self.variables:
            if var.position is None:
                var.position = (random.random(), random.random())
        for con in self.constraints:
            if con.position is None:
                con.position = tuple(sum(var.position[i] for var in con.scope)/len(con.scope)
                                         for i in range(2))
            bbox = dict(boxstyle="square,pad=1.0",color="green")
            for var in con.scope:
                ax.annotate(con.string, var.position, xytext=con.position,
                                    arrowprops={'arrowstyle':'-'},bbox=con_bbox,
                                    ha='center')
        for var in self.variables:
            x,y = var.position
            plt.text(x,y,var.name,bbox=var_bbox,ha='center')


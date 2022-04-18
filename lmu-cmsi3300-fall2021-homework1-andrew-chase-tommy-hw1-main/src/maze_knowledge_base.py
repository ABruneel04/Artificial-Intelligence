'''
maze_knowledge_base.py

Specifies a simple, Conjunctive Normal Form Propositional
Logic Knowledge Base for use in Grid Maze pathfinding problems
with side-information.
'''
from maze_clause import MazeClause
import unittest
import copy

class MazeKnowledgeBase:

    def __init__ (self):
        self.clauses = set()

    def tell (self, clause):
        if not self.ask(self.negationHelper(clause)):
            self.clauses.add(clause)
        else:
            print("@@@@@@@@@@@@@@@@@@@@")
            print(clause.props)
            print("@@@@@@@@@@@@@@@@@@@@")
        return
        """
        Adds the given clause to the CNF MazeKnowledgeBase
        Note: we expect that no clause added this way will ever
        make the KB inconsistent (you need not check for this)
        """
        """        key = 0
        while key == 0:
            for j in clause.props:
                if self.ask(MazeClause([(j, not clause.get_prop(j))])):
                    clause.props.pop(j)
                    break
            key = 1
        """

        # TODO: This is currently implemented incorrectly; see
        # spec for details!

    def ask (self, query):
        """
        Given a MazeClause query, returns True if the KB entails
        the query, False otherwise

        for i in self.clauses:
            for j in query.props:
                if i.get_prop(j) == query.get_prop(j):
                    return True
        """
        return self.PLReso(query)
        # TODO: Implement resolution inference here!
        # This is currently implemented incorrectly; see
        # spec for details!

    def negationHelper(self, query):
        for i in query.props:
            query.props[i] = not query.props[i]
        return query

    def PLReso(self, prop):
        clauses = copy.deepcopy(self.clauses)
        clauses.add(self.negationHelper(prop))
        new = set()
        resolvants = set()
        while True:
            for i in clauses:
                for j in clauses:
                    if i != j:
                        resolvant = MazeClause.resolve(i,j)
                        resolvants.update(resolvant)
                        if len(resolvant) == 1:
                            if list(resolvant)[0].is_empty():
                                return True
                        new = new.union(resolvants)
            if new.issubset(clauses):
                return False
            clauses = clauses.union(new)

class MazeKnowledgeBaseTests(unittest.TestCase):
    def test_mazekb1(self):
        kb = MazeKnowledgeBase()
        kb.tell(MazeClause([(("X", (1, 1)), True)]))
        self.assertTrue(kb.ask(MazeClause([(("X", (1, 1)), True)])))

    def test_mazekb2(self):
        kb = MazeKnowledgeBase()
        kb.tell(MazeClause([(("X", (1, 1)), False)]))
        kb.tell(MazeClause([(("X", (1, 1)), True), (("Y", (1, 1)), True)]))
        self.assertTrue(kb.ask(MazeClause([(("Y", (1, 1)), True)])))

    def test_mazekb3(self):
        kb = MazeKnowledgeBase()
        kb.tell(MazeClause([(("X", (1, 1)), False), (("Y", (1, 1)), True)]))
        kb.tell(MazeClause([(("Y", (1, 1)), False), (("Z", (1, 1)), True)]))
        kb.tell(MazeClause([(("W", (1, 1)), True), (("Z", (1, 1)), False)]))
        kb.tell(MazeClause([(("X", (1, 1)), True)]))
        self.assertTrue(kb.ask(MazeClause([(("W", (1, 1)), True)])))
        self.assertFalse(kb.ask(MazeClause([(("Y", (1, 1)), False)])))


if __name__ == "__main__":
    unittest.main()

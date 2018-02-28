
class Node(object):
    def __init__(self, id, name):
        self.id = id
        self.name = name
        self.adj_nodes = dict()

        self.rank = 1
        self.c = 0
        self.adj2 = list()

    def addEdge(self, node, cost):
        self.adj_nodes[node.id] = [node, cost]
        self.c += 1
        node.adj2.append(self)

    def toString(self):
        string = str(self.id) + ", " + self.name + "\n"
        #for i in self.adj_nodes.values():
            #string += "\t-> " + str(i[0].id) + ", " + i[0].name
        return string


class Graph(object):
    def __init__(self):
        self.nodes = dict()
        self.size = 0
        self.h_rank = 0
        self.l_rank = 0
        self.rank = list()

    def addNode(self, node):
        self.nodes[node.id] = node
        self.size += 1

    def breadthFirstSearch(self, root):
        from collections import deque
        import networkx as nx
        from networkx.drawing.nx_agraph import write_dot, graphviz_layout
        import matplotlib.pyplot as plt

        G = nx.DiGraph()
        G.add_node(root)

        visited_nodes = list()
        visited_nodes.append(root)
        queue = deque([root])
        tree = dict()

        while len(queue) > 0:
            node = queue.popleft()
            adj_nodes = self.nodes[node].adj_nodes.keys()
            remaining_elements = set(adj_nodes).difference(set(visited_nodes))
            tree[node] = list(remaining_elements)
            if len(remaining_elements) > 0:
                for elem in sorted(remaining_elements):
                    G.add_node(elem)
                    G.add_edge(node, elem)
                    visited_nodes.append(elem)
                    queue.append(elem)

        write_dot(G, 'bfs_tree.dot')
        plt.title('BFS TREE')
        pos = graphviz_layout(G, prog='dot')
        nx.draw(G, pos, with_labels=False, arrows=True)
        plt.savefig('bfs_tree.png')

    def depthFirstSearch(self, root):
        import networkx as nx
        from networkx.drawing.nx_agraph import write_dot, graphviz_layout
        import matplotlib.pyplot as plt

        G = nx.DiGraph()
        G.add_node(root)

        tree = dict()
        visited = list()
        stack = list()
        stack.append(root)
        node = root

        while len(stack) > 0:
            if node not in visited:
                visited.append(node)
                tree[node] = list()

            adj_nodes = self.nodes[node].adj_nodes.keys()

            if set(adj_nodes).issubset(set(visited)):
                stack.pop()
                if len(stack) > 0:
                    node = stack[-1]
                continue
            else:
                remaining_elements = set(adj_nodes).difference(set(visited))

            first_adj_node = sorted(remaining_elements)[0]
            stack.append(first_adj_node)
            G.add_node(first_adj_node)
            G.add_edge(node, first_adj_node)
            tree[node].append(first_adj_node)
            node = first_adj_node


        write_dot(G, 'dfs_tree.dot')
        plt.title('DFS TREE')
        pos = graphviz_layout(G, prog='dot')
        nx.draw(G, pos, with_labels=False, arrows=True)
        plt.savefig('dfs_tree.png')

        return tree

    def pageRank(self, d, iter):
        for i in range(iter):
            for node in self.nodes.itervalues():
                sum = 0
                for item in node.adj2:
                    sum += item.rank / item.c
                node.rank = (1-d) + d*sum

        for node in self.nodes.itervalues():
            self.rank.append(node)

        self.rank.sort(key = lambda x: x.rank, reverse = True)


if __name__ == "__main__":
    import pandas as pd

    network = Graph()
    nodes = pd.read_csv("nodes.csv")
    edges = pd.read_csv("edges.csv")

    for i, row in nodes.iterrows():
        network.addNode(Node(row['Id'], row['Label']))

    for i, row in edges.iterrows():
        network.nodes[row['Source']].addEdge(network.nodes[row['Target']], row['Weight'])

    #tree = (network.depthFirstSearch(153080620724))

    network.pageRank(.85, 1000)

    print network.rank[0].toString()
    print network.rank[-1].toString()


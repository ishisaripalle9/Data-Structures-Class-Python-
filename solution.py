"""
Project 7
CSE 331 S23 (Onsay)
solution.py
"""
import math
import queue
from typing import TypeVar, Generator, List, Tuple, Optional
from collections import Counter, deque
import re
import json

# for more information on typehinting, check out https://docs.python.org/3/library/typing.html
T = TypeVar("T")  # represents generic type
# represents a Node object (forward-declare to use in Node __init__)
Node = TypeVar("Node")

####################################################################################################


class Node:
    """
    Implementation of an AVL tree node.
    Do not modify.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["value", "parent", "left", "right", "height", "data"]

    def __init__(self, value: T, parent: Node = None,
                 left: Node = None, right: Node = None, data: List[str] = None) -> None:
        """
        Construct an AVL tree node.

        :param value: value held by the node object
        :param parent: ref to parent node of which this node is a child
        :param left: ref to left child node of this node
        :param right: ref to right child node of this node
        """
        self.value = value
        self.parent, self.left, self.right = parent, left, right
        self.height = 0
        self.data = data

    def __repr__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return f"<{str(self.value)}>"

    def __str__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return repr(self)


####################################################################################################

class AVLTree:
    """
    Implementation of an AVL tree.
    Modify only below indicated line.
    """

    __slots__ = ["origin", "size"]

    def __init__(self) -> None:
        """
        Construct an empty AVL tree.
        """
        self.origin = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the BSTree as a string.

        :return: string representation of the BST tree
        """
        if self.origin is None:
            return "Empty AVL Tree"

        lines = pretty_print_binary_tree(self.origin, 0, False, '-')[0]
        return "\n" + "\n".join((line.rstrip() for line in lines))

    def __str__(self) -> str:
        """
        Represent the AVL tree as a string.

        :return: string representation of the BST tree
        """
        return repr(self)

    def visualize(self, filename="avl_tree_visualization.svg"):
        """
        Generates an svg image file of the AVL tree.

        :param filename: The filename for the generated svg file. Should end with .svg.
        Defaults to output.svg
        """
        svg_string = svg(self.origin, node_radius=20)
        with open(filename, 'w') as f:
            print(svg_string, file=f)  # This is the line that creates the file in the file system.
        return svg_string

    def height(self, root: Node) -> int:
        """
        Return height of a subtree in the AVL, properly handling the case of root = None.
        Recall that the height of an empty subtree is -1.

        :param root: root node of subtree to be measured
        :return: height of subtree rooted at `root` parameter
        """
        return root.height if root is not None else -1

    def left_rotate(self, root: Node) -> Optional[Node]:
        """
        Perform a left rotation on the subtree rooted at `root`. Return new subtree root.

        :param root: root node of unbalanced subtree to be rotated.
        :return: new root node of subtree following rotation.
        """
        if root is None:
            return None

        # pull right child up and shift right-left child across tree, update parent
        new_root, rl_child = root.right, root.right.left
        root.right = rl_child
        if rl_child is not None:
            rl_child.parent = root

        # right child has been pulled up to new root -> push old root down left, update parent
        new_root.left = root
        new_root.parent = root.parent
        if root.parent is not None:
            if root is root.parent.left:
                root.parent.left = new_root
            else:
                root.parent.right = new_root
        root.parent = new_root

        # handle tree origin case
        if root is self.origin:
            self.origin = new_root

        # update heights and return new root of subtree
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        new_root.height = 1 + \
                          max(self.height(new_root.left), self.height(new_root.right))
        return new_root

    def remove(self, root: Node, val: T) -> Optional[Node]:
        """
        Remove the node with `value` from the subtree rooted at `root` if it exists.
        Return the root node of the balanced subtree following removal.

        :param root: root node of subtree from which to remove.
        :param val: value to be removed from subtree.
        :return: root node of balanced subtree.
        """
        # handle empty and recursive left/right cases
        if root is None:
            return None
        elif val < root.value:
            root.left = self.remove(root.left, val)
        elif val > root.value:
            root.right = self.remove(root.right, val)
        else:
            # handle actual deletion step on this root
            if root.left is None:
                # pull up right child, set parent, decrease size, properly handle origin-reset
                if root is self.origin:
                    self.origin = root.right
                if root.right is not None:
                    root.right.parent = root.parent
                self.size -= 1
                return root.right
            elif root.right is None:
                # pull up left child, set parent, decrease size, properly handle origin-reset
                if root is self.origin:
                    self.origin = root.left
                if root.left is not None:
                    root.left.parent = root.parent
                self.size -= 1
                return root.left
            else:
                # two children: swap with predecessor and delete predecessor
                predecessor = self.max(root.left)
                root.value = predecessor.value
                root.left = self.remove(root.left, predecessor.value)

        # update height and rebalance every node that was traversed in recursive deletion
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        return self.rebalance(root)

    ########################################
    # Implement functions below this line. #
    ########################################

    def right_rotate(self, root: Node) -> Optional[Node]:
        """
        Perform a right rotation on the subtree rooted at `root`. Return new subtree root.
        param root: root node of unbalanced subtree to be rotated.
        return: new root node of subtree following rotation.
        """
        if root is None:
            return None

        # pull left child up and shift left-right child across tree, update parent
        new_root, lr_child = root.left, root.left.right
        root.left = lr_child
        if lr_child is not None:
            lr_child.parent = root

        # left child has been pulled up to new root -> push old root down left, update parent
        new_root.right = root
        new_root.parent = root.parent
        if root.parent is not None:
            if root is root.parent.right:
                root.parent.right = new_root
            else:
                root.parent.left = new_root
        root.parent = new_root

        # handle tree origin case
        if root is self.origin:
            self.origin = new_root

        # update heights and return new root of subtree
        root.height = 1 + max(self.height(root.right), self.height(root.left))
        new_root.height = 1 + \
                          max(self.height(new_root.right), self.height(new_root.left))
        return new_root

    def balance_factor(self, root: Node) -> int:
        """
        Compute the balance factor of the subtree rooted at root
        params: the root Node of the subtree on which to compute the balance factor
        returns: int representing the balance factor of root
        """
        if root is None:
            return 0
        else:
            balance_factor = self.height(root.left) - self.height(root.right)
            return balance_factor

    def rebalance(self, root: Node) -> Optional[Node]:
        """
        Re-balance the subtree rooted at root (if necessary)
        and return the new root of the resulting subtree
        params: the root Node of the subtree to be rebalanced
        returns: root of new subtree after re-balancing (could be the original root)
        """
        if root is None:
            return root
        balance_factor = self.balance_factor(root)
        balance_factor_left = self.balance_factor(root.left)
        balance_factor_right = self.balance_factor(root.right)
        if balance_factor > 1:
            if balance_factor_left >= 0:
                return self.right_rotate(root)
            else:
                root.left = self.left_rotate(root.left)
                return self.right_rotate(root)
        if balance_factor < -1:
            if balance_factor_right <= 0:
                return self.left_rotate(root)
            else:
                root.right = self.right_rotate(root.right)
                return self.left_rotate(root)
        return root

    def insert(self, root: Node, val: T, data: List[str] = None) -> Optional[Node]:
        """
        Insert a node with val into the subtree rooted at root,
        returning the root node of the balanced subtree after insertion
        params: the root Node of the subtree in which to insert val
        returns: root of new subtree after insertion and re-balancing (could be the original root)
        """
        # 1: Origin being none
        if root is None:
            root = Node(value=val, data=data)
            self.size += 1
            self.origin = root
            return root
        # 2: root.val == val we are looking for
        if val == root.value:
            return root
        # 3. Value we are looking for is > than root.val
        # 3a. If no child to that side then create child, assigns value, data AND parent
        if val < root.value:
            if root.left is not None:
                root.left = self.insert(root.left, val, data)
            else:
                root.left = Node(value=val, data=data, parent=root.left)
                self.size += 1
        # 4a,b Same as 3 just on other side
        if val > root.value:
            if root.right is not None:
                root.right = self.insert(root.right, val, data)
            else:
                root.right = Node(value=val, data=data, parent=root.right)
                self.size += 1
        root.height = 1 + max(self.height(root.right), self.height(root.left))
        return self.rebalance(root)

    def min(self, root: Node) -> Optional[Node]:
        """
        Find and return the Node with the smallest
        value in the subtree rooted at root
        params: The root Node of the subtree in which to search for a minimum
        returns: node object containing the smallest value in the subtree rooted at root
        """
        if root is None:
            return root
        else:
            if root.left is None:
                return root
            else:
                return self.min(root.left)

    def max(self, root: Node) -> Optional[Node]:
        """
        Find and return the Node with the largest
        value in the subtree rooted at root
        params: The root Node of the subtree in which to search for a maximum
        returns: node object containing the largest value in the subtree rooted at root
        """
        if root is None:
            return root
        else:
            if root.right is None:
                return root
            else:
                return self.max(root.right)

    def search(self, root: Node, val: T) -> Optional[Node]:
        """
        Find and return the Node with the value val in the subtree rooted at root
        params: the root Node of the subtree in which to search for val
        returns: node object containing val if it exists, else the Node object
        below which val would be inserted as a child
        """
        if root is None:
            return root
        else:
            if val == root.value:
                return root
            else:
                if val < root.value:
                    if root.left:
                        return self.search(root.left, val)
                    else:
                        return root
                elif val > root.value:
                    if root.right:
                        return self.search(root.right, val)
                    else:
                        return root

    def inorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Perform an inorder (left, current, right) traversal of the
        subtree rooted at root using a Python generator
        params: the root Node of the subtree currently being traversed
        returns: Generator object which yields Node objects only (no None-type yields)
        """
        if root is None:
            return
        else:
            yield from self.inorder(root.left)
            yield root
            yield from self.inorder(root.right)

    def __iter__(self) -> Generator[Node, None, None]:
        """
         We want the iteration to use the inorder traversal of the tree so
         this should be implemented such that it returns the inorder traversal
         params: none
         returns: a generator that iterates over the inorder traversal of the tree
        """
        return self.inorder(self.origin)

    def preorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Perform an inorder ( current, left, right) traversal of the
        subtree rooted at root using a Python generator
        params: the root Node of the subtree currently being traversed
        returns: Generator object which yields Node objects only (no None-type yields)
        """
        if root is None:
            return
        else:
            yield root
            yield from self.preorder(root.left)
            yield from self.preorder(root.right)

    def postorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Perform an inorder ( current, right, left) traversal of the
        subtree rooted at root using a Python generator
        params: the root Node of the subtree currently being traversed
        returns: Generator object which yields Node objects only (no None-type yields)
        """
        if root is None:
            return
        else:
            yield from self.postorder(root.left)
            yield from self.postorder(root.right)
            yield root

    def levelorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Perform a level-order (breadth-first) traversal of the
        subtree rooted at root using a Python generator
        params: the root Node of the subtree currently being traversed.
        returns: generator object which yields Node objects only (no None-type yields).
        """
        if root is None:
            return
        else:
            q = [root]
            while q:
                node = q.pop(0)
                yield node
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)

    ####################################################################################################

# ------- DO NOT MODIFY -- USED FOR TESTING ------- #
def is_avl_tree(node):

    def is_avl_tree_inner(cur, high, low):
        # if node is None at any time, it is balanced and therefore True
        if cur is None:
            return True, -1
        if cur.value > high or cur.value < low:
            return False, -1
        is_avl_left, left_height = is_avl_tree_inner(cur.left, cur.value, low)
        is_avl_right, right_height = is_avl_tree_inner(cur.right, high, cur.value)
        cur_height = max(left_height, right_height) + 1
        return is_avl_left and is_avl_right and abs(left_height - right_height) < 2, cur_height

    # absolute difference between right and left subtree should be no greater than 1
    return is_avl_tree_inner(node, float('inf'), -float('inf'))[0]


# ------- APPLICATION PROBLEM ------- #
# ------- DO NOT ALTER ANYTHING BEFORE THE insert_rows FUNCTION ------- #
class Table:
    """
    Table class containing attribute name index map, AVL tree for data, and current insertion index (never decreases).
    """
    __slots__ = ["names_to_indices", "header_row", "data_tree", "insertion_index"]

    def __init__(self, attribute_names: List[str]) -> None:
        """
        Creates a Table using attributes
        """
        self.names_to_indices = {attribute_names[i]: i for i in range(len(attribute_names))}
        self.header_row = attribute_names
        self.header_row.insert(0, "index")
        self.data_tree = AVLTree()
        self.insertion_index = 0

    # ------- COMPLETE THE FOLLOWING FUNCTIONS ------- #
    def insert_rows(self, values: List[List[str]], attributes: List[str]) -> None:
        """
        Goes through the lists in values and inserts their data one by
        one into the Table by inserting into data_tree
        params: values, attributes
        returns: none
        """
        for val in values:
            row = [None] * len(val)
            for i in range(len(val)):
                row[self.names_to_indices[attributes[i]]] = val[i]
            self.data_tree.insert(self.data_tree.origin, self.insertion_index, row)
            self.insertion_index += 1

    def remove_rows(self, indices: List[str]) -> None:
        """
        Removes all specified indices from the table
        params: indices
        returns: none
        """
        for index in indices:
            self.data_tree.remove(self.data_tree.origin, int(index))

    def show_latest(self) -> List[List[str]]:
        """
        Displays the header row and latest (highest index–what function gives you
        the Node with the highest value?) row in a list of lists
        params: none
        returns:  list of lists where the first list is the header row and the second list is the
        values of the most recently inserted row corresponding to the attributes at the same
        indices within the header row
        """
        result = self.header_row
        node = self.data_tree.max(self.data_tree.origin)
        if not self.data_tree.origin:
            return [result]
        else:
            if node is None:
                return [result]
            else:
                data = [node.value]
                data.extend(node.data)
                return [result, data]

    def show_oldest(self) -> List[List[str]]:
        """
        Displays the header row and oldest (lowest index–what function gives you
        the Node with the lowest value?) row in a list of lists
        params: none
        returns: list of lists where the first list is the header row and the second list is the
        values of the oldest inserted row corresponding to the attributes at the same
        indices within the header row
        """
        result = self.header_row
        node = self.data_tree.min(self.data_tree.origin)
        if not self.data_tree.origin:
            return [result]
        else:
            if node is None:
                return [result]
            else:
                data = [node.value]
                data.extend(node.data)
                return [result, data]

    def show_everything(self) -> List[List[str]]:
        """
        Displays the header row and all data rows with their
        indices in ascending index order
        params: none
        returns: List of lists where the first list is the header row and
        the subsequent lists are the rows in the table in ascending index order whose
        values correspond via matching indices to the header row’s attribute names
        """
        result = self.header_row
        if not self.data_tree.origin:
            return [result]
        else:
            index = []
            for row in self.data_tree.inorder(self.data_tree.origin):
                data = [row.value]
                data.extend(row.data)
                index.append(data)
            return [result] + index

# DO NOT MODIFY THIS CLASS #
class AVLDatabase:
    """
    Database class containing table name to Table object map as well as all database methods.
    """
    __slots__ = ["names_to_tables"]

    def __init__(self) -> None:
        """
        Constructs an empty AVL database
        """
        self.names_to_tables = {}

    def query(self, query: str) -> List[List[str]]:  # (or None)
        """
        Performs queries on the database
        :param query: Query in the form of a string
        :returns: List of lists of strings for "SHOW ME" queries, None otherwise
        """
        try:
            query = query.lower()  # all data will be lowercase to keep things simple
            comma_indices = [i for i, ch in enumerate(query) if ch == ',']
            table_name = query[comma_indices[0] + 2:comma_indices[1]]
            # create case
            if query.find("create") != -1:
                # parse query
                attributes_loc = query.find("attributes")
                period_loc = query.rfind('.')
                attributes = query[attributes_loc+11:period_loc].split(', ')

                # create table (unlike SQL, this will overwrite an existing table)
                self.names_to_tables[table_name] = Table(attributes)

            # insert case
            elif query.find("insert into") != -1:
                attributes_loc = query.find("attributes")
                period_loc = query.rfind('.')
                attributes = query[attributes_loc + 11:period_loc].split(', ')

                values_loc = query.find("values")
                with_loc = query.find("with")

                # list of lists of values
                values = [i.split(', ') for i in query[values_loc + 8:with_loc - 1].split('; ')]

                # insert the rows
                self.names_to_tables[table_name].insert_rows(values, attributes)

            # remove case
            elif query.find("remove") != -1:
                indices_loc = query.find("indices")
                period_loc = query.rfind('.')
                indices = query[indices_loc + 8:period_loc].split(', ')

                # remove the rows by their indices
                self.names_to_tables[table_name].remove_rows(indices)

            # show me case
            elif query.find("show me") != -1:
                # show me latest
                if query.rfind("latest") != -1:
                    return self.names_to_tables[table_name].show_latest()
                # show me oldest
                elif query.rfind("oldest") != -1:
                    return self.names_to_tables[table_name].show_oldest()
                # show me everything in ascending order of index
                elif query.rfind("everything") != -1:
                    return self.names_to_tables[table_name].show_everything()

        except KeyError:
            raise KeyError("Something went wrong when performing that operation. "
                           "Check to see if your syntax is valid and the table exists.")

_SVG_XML_TEMPLATE = """
<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
<style>
    .value {{
        font: 300 16px monospace;
        text-align: center;
        dominant-baseline: middle;
        text-anchor: middle;
    }}
    .dict {{
        font: 300 16px monospace;
        dominant-baseline: middle;
    }}
    .node {{
        fill: lightgray;
        stroke-width: 1;
    }}
</style>
<g stroke="#000000">
{body}
</g>
</svg>
"""

_NNC_DICT_BOX_TEXT_TEMPLATE = """<text class="dict" y="{y}" xml:space="preserve">
    <tspan x="{label_x}" dy="1.2em">{label}</tspan>
    <tspan x="{bracket_x}" dy="1.2em">{{</tspan>
    {values}
    <tspan x="{bracket_x}" dy="1.2em">}}</tspan>
</text>
"""


def pretty_print_binary_tree(root: Node, curr_index: int, include_index: bool = False,
                             delimiter: str = "-", ) -> \
        Tuple[List[str], int, int, int]:
    """
    Taken from: https://github.com/joowani/binarytree

    Recursively walk down the binary tree and build a pretty-print string.
    In each recursive call, a "box" of characters visually representing the
    current (sub)tree is constructed line by line. Each line is padded with
    whitespaces to ensure all lines in the box have the same length. Then the
    box, its width, and start-end positions of its root node value repr string
    (required for drawing branches) are sent up to the parent call. The parent
    call then combines its left and right sub-boxes to build a larger box etc.
    :param root: Root node of the binary tree.
    :type root: binarytree.Node | None
    :param curr_index: Level-order_ index of the current node (root node is 0).
    :type curr_index: int
    :param include_index: If set to True, include the level-order_ node indexes using
        the following format: ``{index}{delimiter}{value}`` (default: False).
    :type include_index: bool
    :param delimiter: Delimiter character between the node index and the node
        value (default: '-').
    :type delimiter:
    :return: Box of characters visually representing the current subtree, width
        of the box, and start-end positions of the repr string of the new root
        node value.
    :rtype: ([str], int, int, int)
    .. _Level-order:
        https://en.wikipedia.org/wiki/Tree_traversal#Breadth-first_search
    """
    if root is None:
        return [], 0, 0, 0

    line1 = []
    line2 = []
    if include_index:
        node_repr = "{}{}{}".format(curr_index, delimiter, root.value)
    else:
        node_repr = f'{root.value},h={root.height},' \
                    f'⬆{str(root.parent.value) if root.parent else "None"}'

    new_root_width = gap_size = len(node_repr)

    # Get the left and right sub-boxes, their widths, and root repr positions
    l_box, l_box_width, l_root_start, l_root_end = pretty_print_binary_tree(
        root.left, 2 * curr_index + 1, include_index, delimiter
    )
    r_box, r_box_width, r_root_start, r_root_end = pretty_print_binary_tree(
        root.right, 2 * curr_index + 2, include_index, delimiter
    )

    # Draw the branch connecting the current root node to the left sub-box
    # Pad the line with whitespaces where necessary
    if l_box_width > 0:
        l_root = (l_root_start + l_root_end) // 2 + 1
        line1.append(" " * (l_root + 1))
        line1.append("_" * (l_box_width - l_root))
        line2.append(" " * l_root + "/")
        line2.append(" " * (l_box_width - l_root))
        new_root_start = l_box_width + 1
        gap_size += 1
    else:
        new_root_start = 0

    # Draw the representation of the current root node
    line1.append(node_repr)
    line2.append(" " * new_root_width)

    # Draw the branch connecting the current root node to the right sub-box
    # Pad the line with whitespaces where necessary
    if r_box_width > 0:
        r_root = (r_root_start + r_root_end) // 2
        line1.append("_" * r_root)
        line1.append(" " * (r_box_width - r_root + 1))
        line2.append(" " * r_root + "\\")
        line2.append(" " * (r_box_width - r_root))
        gap_size += 1
    new_root_end = new_root_start + new_root_width - 1

    # Combine the left and right sub-boxes with the branches drawn above
    gap = " " * gap_size
    new_box = ["".join(line1), "".join(line2)]
    for i in range(max(len(l_box), len(r_box))):
        l_line = l_box[i] if i < len(l_box) else " " * l_box_width
        r_line = r_box[i] if i < len(r_box) else " " * r_box_width
        new_box.append(l_line + gap + r_line)

    # Return the new box, its width and its root repr positions
    return new_box, len(new_box[0]), new_root_start, new_root_end


def svg(root: Node, node_radius: int = 16, nnc_mode=False) -> str:
    """
    Taken from: https://github.com/joowani/binarytree

    Generate SVG XML.
    :param root: Generate SVG for tree rooted at root
    :param node_radius: Node radius in pixels (default: 16).
    :type node_radius: int
    :return: Raw SVG XML.
    :rtype: str
    """
    tree_height = root.height
    scale = node_radius * 3
    xml = deque()
    nodes_for_nnc_visualization = []

    def scale_x(x: int, y: int) -> float:
        diff = tree_height - y
        x = 2 ** (diff + 1) * x + 2 ** diff - 1
        return 1 + node_radius + scale * x / 2

    def scale_y(y: int) -> float:
        return scale * (1 + y)

    def add_edge(parent_x: int, parent_y: int, node_x: int, node_y: int) -> None:
        xml.appendleft(
            '<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>'.format(
                x1=scale_x(parent_x, parent_y),
                y1=scale_y(parent_y),
                x2=scale_x(node_x, node_y),
                y2=scale_y(node_y),
            )
        )

    def add_node(node_x: int, node_y: int, node: Node) -> None:
        x, y = scale_x(node_x, node_y), scale_y(node_y)
        xml.append(f'<circle class="node" cx="{x}" cy="{y}" r="{node_radius}"/>')

        if nnc_mode:
            nodes_for_nnc_visualization.append(node.value)
            xml.append(f'<text class="value" x="{x}" y="{y + 5}">key={node.value.key}</text>')
        else:
            xml.append(f'<text class="value" x="{x}" y="{y + 5}">{node.value}</text>')

    current_nodes = [root.left, root.right]
    has_more_nodes = True
    y = 1

    add_node(0, 0, root)

    while has_more_nodes:

        has_more_nodes = False
        next_nodes: List[Node] = []

        for x, node in enumerate(current_nodes):
            if node is None:
                next_nodes.append(None)
                next_nodes.append(None)
            else:
                if node.left is not None or node.right is not None:
                    has_more_nodes = True

                add_edge(x // 2, y - 1, x, y)
                add_node(x, y, node)

                next_nodes.append(node.left)
                next_nodes.append(node.right)

        current_nodes = next_nodes
        y += 1

    svg_width = scale * (2 ** tree_height)
    svg_height = scale * (2 + tree_height)
    if nnc_mode:

        line_height = 20
        box_spacing = 10
        box_margin = 5
        character_width = 10

        max_key_count = max(map(lambda obj: len(obj.dictionary), nodes_for_nnc_visualization))
        box_height = (max_key_count + 3) * line_height + box_margin

        def max_length_item_of_node_dict(node):
            # Check if dict is empty so max doesn't throw exception
            if len(node.dictionary) > 0:
                item_lengths = map(lambda pair: len(str(pair)), node.dictionary.items())
                return max(item_lengths)
            return 0

        max_value_length = max(map(max_length_item_of_node_dict, nodes_for_nnc_visualization))
        box_width = max(max_value_length * character_width, 110)

        boxes_per_row = svg_width // box_width
        rows_needed = math.ceil(len(nodes_for_nnc_visualization) / boxes_per_row)

        nodes_for_nnc_visualization.sort(key=lambda node: node.key)
        for index, node in enumerate(nodes_for_nnc_visualization):
            curr_row = index // boxes_per_row
            curr_column = index % boxes_per_row

            box_x = curr_column * (box_width + box_spacing)
            box_y = curr_row * (box_height + box_spacing) + svg_height
            box = f'<rect x="{box_x}" y="{box_y}" width="{box_width}" ' \
                  f'height="{box_height}" fill="white" />'
            xml.append(box)

            value_template = '<tspan x="{value_x}" dy="1.2em">{key}: {value}</tspan>'
            text_x = box_x + 10

            def item_pair_to_svg(pair):
                return value_template.format(key=pair[0], value=pair[1], value_x=text_x + 10)

            values = map(item_pair_to_svg, node.dictionary.items())
            text = _NNC_DICT_BOX_TEXT_TEMPLATE.format(
                y=box_y,
                label=f"key = {node.key}",
                label_x=text_x,
                bracket_x=text_x,
                values='\n'.join(values)
            )
            xml.append(text)

        svg_width = boxes_per_row * (box_width + box_spacing * 2)
        svg_height += rows_needed * (box_height + box_spacing * 2)

    return _SVG_XML_TEMPLATE.format(
        width=svg_width,
        height=svg_height,
        body="\n".join(xml),
    )


if __name__ == "__main__":
    pass

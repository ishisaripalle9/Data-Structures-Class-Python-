from typing import TypeVar          # For use in type hinting

# Type Declarations
T = TypeVar('T')        # generic type
SLL = TypeVar('SLL')    # forward declared
Node = TypeVar('Node')  # forward declare `Node` type


class SLLNode:
    """
    Node implementation
    Do not modify.
    """

    __slots__ = ['val', 'next']

    def __init__(self, value: T, next: Node = None) -> None:
        """
        Initialize an SLL Node
        :param value: value held by node
        :param next: reference to the next node in the SLL
        :return: None
        """
        self.val = value
        self.next = next

    def __str__(self) -> str:
        """
        Overloads `str()` method to casts nodes to strings
        return: string
        """
        return '(Node: ' + str(self.val) + ' )'

    def __repr__(self) -> str:
        """
        Overloads `repr()` method for use in debugging
        return: string
        """
        return '(Node: ' + str(self.val) + ' )'

    def __eq__(self, other: Node) -> bool:
        """
        Overloads `==` operator to compare nodes
        :param other: right operand of `==`
        :return: bool
        """
        return self is other if other is not None else False


class SinglyLinkedList:
    """
    Implementation of an SLL
    """

    __slots__ = ['head', 'tail']

    def __init__(self) -> None:
        """
        Initializes an SLL
        :return: None
        DO NOT MODIFY THIS FUNCTION
        """
        self.head = None
        self.tail = None

    def __repr__(self) -> str:
        """
        Represents an SLL as a string
        DO NOT MODIFY THIS FUNCTION
        """
        return self.to_string()

    def __eq__(self, other: SLL) -> bool:
        """
        Overloads `==` operator to compare SLLs
        :param other: right hand operand of `==`
        :return: `True` if equal, else `False`
        DO NOT MODIFY THIS FUNCTION
        """
        comp = lambda n1, n2: n1 == n2 and (comp(n1.next, n2.next) if (n1 and n2) else True)
        return comp(self.head, other.head)

# ============ Modify below ============ #
    def push(self, value: T) -> None:
        """
        Pushes an SLLNode to the end of the list
        :param value: value to push to the list
        :return: None
        """
        node_new = SLLNode(value)

        if self.head is None and self.tail is None:
            self.head = node_new
            self.tail = node_new
        else:
            self.tail.next = node_new
            self.tail = node_new

    def to_string(self) -> str:
        """
        Converts an SLL to a string
        :return: string representation of the linked list
        """
        current = self.head
        node_to_string = ""

        if self.head is None and self.tail is None:
            return "None"
        elif current is self.head and current is self.tail:
            node_to_string += current.val
            return node_to_string

        while current and current.next:
            node_to_string += str(current.val) + " --> "
            current = current.next
        node_to_string += str(current.val)
        return node_to_string

    def length(self) -> int:
        """
        Determines number of nodes in the list
        :return: number of nodes in list
        """
        current = self.head
        count = 0

        if self.head is None and self.tail is None:
            return count
        elif current is self.head and current is self.tail:
            count += 1
            return count

        while current and current.next:
            count += 1
            current = current.next
        count += 1
        return count

    def sum_list(self) -> T:
        """
        Sums the values in the list
        :return: sum of values in list
        """
        current = self.head

        if self.head is None and self.tail is None:
            return None

        elif type(current) == int:
            total_sum = 0
            if current is self.head and current is self.tail:
                total_sum += current.val
                return total_sum
            else:
                while current and current.next:
                    total_sum += current.val
                    current = current.next
                total_sum += current.val
                return total_sum

        elif type(current) == str:
            total_sum = ""
            if current is self.head and current is self.tail:
                total_sum += current.val
                return total_sum
            else:
                while current and current.next:
                    total_sum += current.val
                    current = current.next
                total_sum += current.val
                return total_sum

    def remove(self, value: T) -> bool:
        """
        Removes the first node containing `value` from the SLL
        :param value: value to remove
        :return: True if a node was removed, False otherwise
        """
        current = self.head
        previous = None

        if self.head is None and self.tail is None:
            return False
        elif self.head is self.tail:
            if self.head.val is value:
                self.head = None
                self.tail = None
                return True
            else:
                return False
        else:
            while current is not self.tail:
                if current.val is value:
                    if not previous:
                        self.head = current.next
                    else:
                        previous.next = current.next
                    return True
                previous = current
                current = current.next
            if current is self.tail:
                if current.val is value:
                    previous.next = current.next
                    self.tail = previous
                    return True
            return False

    def remove_all(self, value: T) -> bool:
        """
        Removes all instances of a node containing `value` from the SLL
        :param value: value to remove
        :return: True if a node was removed, False otherwise
        """
        if self.head is None and self.tail is None:
            return False
        elif self.head is self.tail:
            if self.head.val is value:
                self.head = None
                self.tail = None
                return True
            else:
                return False
        else:
            current = self.head
            previous = None
            count = 0
            while current:
                if current.val is value:
                    if previous is None:
                        self.head = current.next
                    else:
                        previous.next = current.next
                    count += 1
                if current is self.tail:
                    if self.tail.val is value:
                        previous.next = current.next
                        self.tail = previous
                        count += 1
                    break
                previous = current
                current = current.next
            if self.head is not None and self.head.val is value:
                self.head = self.head.next
            if count > 0:
                return True
            return False

    def search(self, value: T) -> bool:
        """
        Searches the SLL for a node containing `value`
        :param value: value to search for
        :return: `True` if found, else `False`
        """
        current = self.head

        if self.head is None and self.tail is None:
            return False
        elif self.head is self.tail:
            if self.head.val is value:
                return True
            else:
                return False
        else:
            while current and current.next:
                if current.val is value:
                    return True
                current = current.next
            if current.val is value:
                return True
            else:
                return False

    def count(self, value: T) -> int:
        """
        Returns the number of occurrences of `value` in this list
        :param value: value to count
        :return: number of time the value occurred
        """
        current = self.head
        count = 0

        if self.head is None and self.tail is None:
            return count
        elif self.head is self.tail:
            if self.head.val is value:
                count += 1
                return count
            else:
                return count
        else:
            while current and current.next:
                if current.val is value:
                    count += 1
                current = current.next
            if current.val is value:
                count += 1
            return count


def show_encrypted(data: SLL) -> None:
    """
    Reverses the SLL
    :param data: an SLL
    :return: None
    """
    if data.head is None:
        data.head = None
    elif data.head.next is None:
        data.head = data.head
    else:
        previous = None
        current = data.head
        data.head = data.tail
        data.tail = current
        while current:
            next_node = current.next
            current.next = previous
            previous = current
            current = next_node


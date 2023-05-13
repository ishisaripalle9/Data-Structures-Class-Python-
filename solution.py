"""
CSE331 Project 6 SS'23
Circular Double-Ended Queue
solution.py
"""
from collections import deque
from typing import TypeVar, List
from random import randint, shuffle
from timeit import default_timer
# from matplotlib import pyplot as plt  # COMMENT OUT THIS LINE (and `plot_speed`) if you dont want matplotlib
import gc


T = TypeVar('T')


class CircularDeque:
    """
    Representation of a Circular Deque using an underlying python list
    """

    __slots__ = ['capacity', 'size', 'queue', 'front', 'back']

    def __init__(self, data: List[T] = None, front: int = 0, capacity: int = 4):
        """
        Initializes an instance of a CircularDeque
        :param data: starting data to add to the deque, for testing purposes
        :param front: where to begin the insertions, for testing purposes
        :param capacity: number of slots in the Deque
        """
        if data is None and front != 0:
            data = ['Start']  # front will get set to 0 by a front enqueue if the initial data is empty
        elif data is None:
            data = []

        self.capacity: int = capacity
        self.size: int = len(data)
        self.queue: List[T] = [None] * capacity
        self.back: int = (self.size + front - 1) % self.capacity if data else None
        self.front: int = front if data else None

        for index, value in enumerate(data):
            self.queue[(index + front) % capacity] = value

    def __str__(self) -> str:
        """
        Provides a string representation of a CircularDeque
        'F' indicates front value
        'B' indicates back value
        :return: the instance as a string
        """
        if self.size == 0:
            return "CircularDeque <empty>"

        str_list = ["CircularDeque <"]
        for i in range(self.capacity):
            str_list.append(f"{self.queue[i]}")
            if i == self.front:
                str_list.append('(F)')
            elif i == self.back:
                str_list.append('(B)')
            if i < self.capacity - 1:
                str_list.append(',')

        str_list.append(">")
        return "".join(str_list)

    __repr__ = __str__

    #
    # Your code goes here!
    #
    def __len__(self) -> int:
        """
        Returns the length/size of the circular deque - this is the number of items
        currently in the circular deque, and will not necessarily be equal to the capacity
        params: none
        returns: int representing length of the circular deque
        """
        return self.size

    def is_empty(self) -> bool:
        """
        Returns a boolean indicating if the circular deque is empty
        params: none
        returns: True if empty, False otherwise
        """
        if self.size > 0:
            return False
        else:
            return True

    def front_element(self) -> T:
        """
        Returns the first element in the circular deque
        params: none
        returns: the first element if it exists, otherwise None
        """
        if self.is_empty():
            return None
        else:
            return self.queue[self.front]

    def back_element(self) -> T:
        """
        Returns the last element in the circular deque
        params: none
        returns: the last element if it exists, otherwise None
        """
        if self.is_empty():
            return None
        else:
            return self.queue[self.back]

    def enqueue(self, value: T, front: bool = True) -> None:
        """
        Add a value to either the front or back of the circular
        deque based off the parameter front
        params: value, front
        returns: none
        """
        if self.is_empty():
            self.front = 0
            self.back = 0
            self.queue[self.front] = value
            self.size += 1
        else:
            if front is True:
                self.front = (self.front - 1) % self.capacity
                self.queue[self.front] = value
            else:
                self.back = (self.back + 1) % self.capacity
                self.queue[self.back] = value
            self.size += 1
            if self.size == self.capacity:
                self.grow()

    def dequeue(self, front: bool = True) -> T:
        """
        Remove an item from the queue
        params: front
        returns: removed item, None if empty
        """
        if self.is_empty():
            return None
        else:
            if front is True:
                value = self.queue[self.front]
                self.front = (self.front + 1) % self.capacity
            else:
                value = self.queue[self.back]
                self.back = (self.back - 1) % self.capacity
            self.size -= 1
            if self.size <= (self.capacity // 4) and (self.capacity // 2) >= 4:
                self.shrink()
            return value

    def grow(self) -> None:
        """
        Doubles the capacity of CD by creating a new underlying python list
        with double the capacity of the old one and copies the values over from the
        current list
        params: none
        returns: none
        """
        new_deque = [None] * (self.capacity * 2)
        for i in range(self.size):
            new_deque[i] = self.queue[(self.front + i) % self.capacity]
        self.queue = new_deque
        self.front = 0
        self.back = len(self) - 1
        self.capacity = self.capacity * 2

    def shrink(self) -> None:
        """
        Cuts the capacity of the queue in half using the same idea as grow.
        Copy over contents of the old list to a new list with half the capacity.
        params: none
        returns: none
        """
        if (self.capacity // 2) < 4:
            return
        else:
            new_deque = [None] * (self.capacity // 2)
            for i in range(self.size):
                new_deque[i] = self.queue[(self.front + i) % self.capacity]
            self.queue = new_deque
            self.front = 0
            self.back = len(self) - 1
            self.capacity = self.capacity // 2


def get_winning_numbers(numbers: List[int], size: int) -> List[int]:
    """
    Takes in a list of numbers and a sliding window size and returns a
    list containing the maximum value of the sliding window at each iteration step
    params: numbers, size
    returns: a list containing the max sliding window at each iteration step
    """
    winning_numbers = []
    window = CircularDeque()
    if not numbers:
        return []
    else:
        for i in range(len(numbers)):
            while window and window.front_element() < numbers[i]:
                window.dequeue()
            window.enqueue(numbers[i])
            winning_numbers.append(window.back_element())
            if numbers[i - size + 1] == window.back_element():
                window.dequeue(front=False)
        return winning_numbers[size-1:]

    
def get_winning_probability(winning_numbers: List[int]) -> int:
    """
    Takes in a list of winning numbers and returns the probability of
    the numbers winning by finding the largest sum of non-adjacent numbers
    params: winning_numbers
    returns: an integer representing the probability of the numbers winning
    """
    if not winning_numbers:
        return 0
    elif 0 < len(winning_numbers) <= 2:
        return max(winning_numbers)
    else:
        i = winning_numbers[0]
        j = 0
        for k in range(1, len(winning_numbers)):
            i, j = j + winning_numbers[k], max(j, i)
        return max(i, j)

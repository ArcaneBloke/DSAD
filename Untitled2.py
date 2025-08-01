#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Stock Price Monitoring System - Core Data Structure Implementation
Team Member 1 - Data Structure Foundation
Author: Team Member 1
Date: July 31, 2025

This module implements the core linked list data structure for storing stock prices.
Provides insert, delete, and traversal operations with proper error handling.
"""

from datetime import datetime

class StockPriceNode:
    """
    Node class for the linked list to store individual stock price data.
    Each node contains price value, timestamp, and reference to next node.
    """

    def __init__(self, price, timestamp=None):
        """
        Initialize a stock price node.

        Args:
            price (float/int): Stock price value
            timestamp (str): Optional timestamp string

        Raises:
            TypeError: If price is not numeric
            ValueError: If price is negative
        """
        if not isinstance(price, (int, float)):
            raise TypeError("ERROR: Price must be a numeric value (int or float)")

        if price < 0:
            raise ValueError("ERROR: Stock price cannot be negative")

        self.price = float(price)
        self.timestamp = timestamp if timestamp else str(datetime.now())
        self.next = None

    def __str__(self):
        """String representation of the node."""
        return f"StockPriceNode(price=${self.price:.2f}, timestamp={self.timestamp})"


class StockPriceLinkedList:
    """
    Linked List implementation specifically designed for stock price storage.
    Provides efficient insertion, deletion, and traversal operations.
    Maintains chronological order with automatic capacity management.
    """

    def __init__(self, max_capacity=1000):
        """
        Initialize the linked list with specified maximum capacity.

        Args:
            max_capacity (int): Maximum number of nodes allowed

        Raises:
            ValueError: If max_capacity is not positive
        """
        if not isinstance(max_capacity, int) or max_capacity <= 0:
            raise ValueError("ERROR: Maximum capacity must be a positive integer")

        self.head = None
        self.tail = None
        self.size = 0
        self.max_capacity = max_capacity

    def is_empty(self):
        """
        Check if the linked list is empty.

        Returns:
            bool: True if empty, False otherwise

        Time Complexity: O(1)
        """
        return self.size == 0

    def is_full(self):
        """
        Check if the linked list is at maximum capacity.

        Returns:
            bool: True if full, False otherwise

        Time Complexity: O(1)
        """
        return self.size >= self.max_capacity

    def get_size(self):
        """
        Get current number of nodes in the linked list.

        Returns:
            int: Current size

        Time Complexity: O(1)
        """
        return self.size

    def get_capacity_info(self):
        """
        Get comprehensive capacity information.

        Returns:
            tuple: (current_size, max_capacity, is_empty, is_full)

        Time Complexity: O(1)
        """
        return (self.size, self.max_capacity, self.is_empty(), self.is_full())

    def insert_at_end(self, price, timestamp=None):
        """
        Insert a new stock price at the end of the linked list (most recent).
        This maintains chronological order with newest prices at the tail.

        Args:
            price (float): Stock price to insert
            timestamp (str): Optional timestamp string

        Returns:
            str: Success message

        Raises:
            OverflowError: If linked list is at maximum capacity
            TypeError/ValueError: From StockPriceNode validation

        Time Complexity: O(1)
        """
        if self.is_full():
            raise OverflowError(
                f"ERROR: Cannot insert - Linked list is at maximum capacity ({self.max_capacity}). "
                f"Current size: {self.size}"
            )

        try:
            new_node = StockPriceNode(price, timestamp)
        except (TypeError, ValueError) as e:
            raise e

        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node

        self.size += 1

        return f"SUCCESS: Stock price ${price:.2f} inserted successfully at position {self.size}"

    def delete_from_beginning(self):
        """
        Delete the oldest stock price from the beginning of the linked list.
        This removes the head node and updates references accordingly.

        Returns:
            tuple: (deleted_price, success_message)

        Raises:
            IndexError: If linked list is empty

        Time Complexity: O(1)
        """
        if self.is_empty():
            raise IndexError("ERROR: Cannot delete - Linked list is empty")

        deleted_price = self.head.price
        deleted_timestamp = self.head.timestamp

        if self.head == self.tail:
            self.head = None
            self.tail = None
        else:
            self.head = self.head.next

        self.size -= 1

        success_message = (
            f"SUCCESS: Oldest stock price ${deleted_price:.2f} "
            f"(timestamp: {deleted_timestamp}) deleted successfully"
        )

        return deleted_price, success_message

    def get_all_prices(self):
        """
        Retrieve all stock prices from the linked list in chronological order.
        Traverses from head (oldest) to tail (newest).

        Returns:
            list: List of all stock prices in chronological order

        Raises:
            IndexError: If linked list is empty

        Time Complexity: O(n) where n is the number of nodes
        """
        if self.is_empty():
            raise IndexError("ERROR: Cannot retrieve prices - Linked list is empty")

        prices = []
        current = self.head

        while current is not None:
            prices.append(current.price)
            current = current.next

        return prices

    def get_recent_prices(self, count):
        """
        Get the most recent 'count' stock prices from the linked list.
        Returns prices in chronological order (oldest to newest among the recent ones).

        Args:
            count (int): Number of recent prices to retrieve

        Returns:
            list: List of recent prices in chronological order

        Raises:
            ValueError: If count is not positive
            IndexError: If linked list is empty

        Time Complexity: O(n) where n is the size of the list
        """
        if not isinstance(count, int) or count <= 0:
            raise ValueError("ERROR: Count must be a positive integer")

        if self.is_empty():
            raise IndexError("ERROR: Cannot retrieve prices - Linked list is empty")

        all_prices = self.get_all_prices()

        if count >= len(all_prices):
            return all_prices
        else:
            return all_prices[-count:]

    def get_price_at_position(self, position):
        """
        Get stock price at a specific position (0-indexed from head).

        Args:
            position (int): Position index (0 = head/oldest)

        Returns:
            float: Stock price at the specified position

        Raises:
            ValueError: If position is invalid
            IndexError: If position is out of bounds or list is empty

        Time Complexity: O(n) in worst case
        """
        if not isinstance(position, int) or position < 0:
            raise ValueError("ERROR: Position must be a non-negative integer")

        if self.is_empty():
            raise IndexError("ERROR: Cannot access position - Linked list is empty")

        if position >= self.size:
            raise IndexError(f"ERROR: Position {position} out of bounds. List size: {self.size}")

        current = self.head
        for i in range(position):
            current = current.next

        return current.price

    def clear_all(self):
        """
        Remove all nodes from the linked list.
        Resets the list to empty state.

        Returns:
            str: Success message

        Time Complexity: O(1)
        """
        nodes_cleared = self.size

        self.head = None
        self.tail = None
        self.size = 0

        return f"SUCCESS: Cleared {nodes_cleared} nodes. Linked list is now empty."

    def __len__(self):
        """
        Return the number of nodes in the linked list.
        Enables use of len() function.
        """
        return self.size

    def __str__(self):
        """
        String representation of the linked list.
        Shows size, capacity, and price range information.
        """
        if self.is_empty():
            return f"StockPriceLinkedList: Empty (0/{self.max_capacity})"

        try:
            all_prices = self.get_all_prices()
            min_price = min(all_prices)
            max_price = max(all_prices)
            latest_price = all_prices[-1]

            return (
                f"StockPriceLinkedList: {self.size}/{self.max_capacity} prices, "
                f"Range: ${min_price:.2f}-${max_price:.2f}, "
                f"Latest: ${latest_price:.2f}"
            )
        except Exception:
            return f"StockPriceLinkedList: {self.size}/{self.max_capacity} prices"

# ==============================================================================
# INTEGRATION INTERFACE FOR OTHER TEAM MEMBERS
# This function is the primary interface for other team members to use
# the data structure implemented by Team Member 1.
# ==============================================================================

def create_stock_price_storage(max_capacity=1000):
    """
    Factory function to create a stock price storage system.
    This is the main interface other team members should use.

    Args:
        max_capacity (int): Maximum number of prices to store

    Returns:
        StockPriceLinkedList: Configured linked list for stock prices

    Usage for Team Members 2 & 3:
        storage = create_stock_price_storage(500)
        storage.insert_at_end(100.50)
        prices = storage.get_recent_prices(5)
    """
    try:
        return StockPriceLinkedList(max_capacity)
    except ValueError as e:
        print(f"Error creating storage: {e}")
        return None


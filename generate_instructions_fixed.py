import json
import random

# Properly matched Q&A pairs
qa_pairs = {
    "What is recursion in programming?": "Recursion is a programming technique where a function calls itself to solve a problem. Each call works on a smaller part of the problem. A base case is required to stop the recursion and prevent infinite calls.",
    "Explain recursion with a simple example.": "In recursion, a problem is broken into smaller subproblems of the same type. The function keeps calling itself until a base condition is met. This approach is useful for problems like tree traversal.",
    "Why is a base case important in recursion?": "A base case is required to stop the recursion and prevent infinite calls. Without it, the function would call itself indefinitely, leading to a stack overflow.",
    "What problems are best solved using recursion?": "Recursion is best suited for problems that can be broken down into similar subproblems, such as tree traversal, factorial calculation, and solving mathematical sequences.",
    "What happens if recursion has no base case?": "Without a base case, recursion leads to infinite function calls, eventually causing a stack overflow error and program crash.",
    
    "What is a stack data structure?": "A stack is a data structure that follows the Last In First Out principle. Elements are added and removed from the top. Stacks are used in function calls and expression evaluation.",
    "Explain stack operations with an example.": "Stacks store data in a linear order where the last inserted element is accessed first. Common operations include push to add elements and pop to remove elements. A stack overflow occurs when memory limit is exceeded.",
    "Where are stacks used in real programs?": "Stacks are used in function call management, expression evaluation, backtracking algorithms, undo mechanisms, and browser history navigation.",
    "What is stack overflow?": "Stack overflow occurs when a stack exceeds its memory limit, typically caused by too many recursive calls or excessive data storage.",
    "How does a stack differ from a queue?": "A stack follows LIFO Last In First Out while a queue follows FIFO First In First Out. Stacks add and remove from one end, queues add at one end and remove from the other.",
    
    "What is a queue data structure?": "A queue is a data structure that follows the First In First Out principle. Elements are inserted at the rear and removed from the front. Queues are used in scheduling and buffering.",
    "Explain FIFO with a real-life example.": "Queues process elements in the order they arrive, like a line at a store. The first person to join the line is the first to be served. This makes them useful in task scheduling and printer queues.",
    "Where are queues used in computing?": "Queues are used in task scheduling, printer spooling, breadth-first search, message buffering, and handling asynchronous data transfers.",
    "Difference between queue and stack.": "A queue follows FIFO First In First Out while a stack follows LIFO Last In First Out. Queues are used for sequential processing, stacks for backtracking.",
    "What is a circular queue?": "A circular queue is a queue where the last position connects back to the first, forming a circle. This allows efficient use of memory by reusing freed positions.",
    
    "What is time complexity?": "Time complexity describes how an algorithm's running time increases with input size. It helps compare algorithm efficiency using Big-O notation.",
    "Why is Big-O notation used?": "Big-O notation provides a standardized way to describe algorithm efficiency by showing the worst-case growth rate of time or space requirements.",
    "Explain O(n) with an example.": "O of n means linear time complexity where execution time grows proportionally with input size. For example, finding an element in an unsorted array requires checking each element once.",
    "What is space complexity?": "Space complexity measures how much memory an algorithm uses as input size grows. Efficient algorithms try to minimize both time and space usage.",
    "Difference between time and space complexity.": "Time complexity measures how long an algorithm takes to run, while space complexity measures how much memory it uses. Both are important for algorithm efficiency.",
    
    "What is a linked list?": "A linked list is a data structure where elements are stored as nodes connected by pointers. Each node contains data and a reference to the next node. Linked lists allow dynamic memory usage.",
    "What is an array?": "An array is a data structure that stores elements of the same type in contiguous memory locations. Elements can be accessed using an index. Arrays provide fast access but have a fixed size.",
    "What is binary search?": "Binary search is an efficient algorithm used on sorted arrays. It works by repeatedly dividing the search space in half. This reduces time complexity to O of log n.",
    "What is object-oriented programming?": "Object-oriented programming is a paradigm based on objects and classes. It focuses on encapsulation, inheritance, polymorphism, and abstraction. OOP helps build modular and reusable code.",
    "What is a variable?": "A variable is a named storage location in memory that holds a value. Variables have a data type and can be changed during program execution.",
}

samples = []

# Generate 500 samples by repeating the Q&A pairs
qa_list = list(qa_pairs.items())
random.seed(42)  # For reproducibility

while len(samples) < 500:
    q, a = random.choice(qa_list)
    samples.append({
        "instruction": q,
        "input": "",
        "output": a
    })

# Save to file
with open("instruction_clean_fixed.json", "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)

print(f"✅ Generated {len(samples)} properly matched instruction samples")
print(f"📝 Unique Q&A pairs: {len(qa_pairs)}")

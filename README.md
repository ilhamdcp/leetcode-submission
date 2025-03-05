# leetcode-submission
A repo that signifies I am being productive at least when I am not developing my personal projects

# Key Points
## Kadane Algorithm
An algorithm that specializes in maximum subarray problem. The logic is the following:

- if the maxSubArray currently negative, it is always better to start a new one
- if the maxSubArray currently positive, it is always better to extend the current subArray
*Note: utilize max function for maximum sub array problems and min otherwise*

## 3Sum Problem
It is an extension of 2Sum problem, where in 2Sum, given any number inside an array, find the other element that satisfy the target, for example number zero. 
2Sum is quite simple (hint: use hashmap). But when it comes to 3Sum, the complexity comes, especially when the ordering of the elements inside the triplets is not considered. Here are the algorithm works
- sort the array ascending
- for each index, iterate while left < right
  - set pointer left = i + 1, right = len(array) - 1
  - if the triplets < target, increment left
  - if the triplets > target, decrement right
  - if triplets = target and not in usedTriplets (a set of tuples can be used to store usedTriplets), add to usedTriplets and add to list of triplets
- return the list of triplets
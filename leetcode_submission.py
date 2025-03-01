# https://leetcode.com/problems/count-the-number-of-consistent-strings/description/

def countConsistentStrings(self, allowed: str, words: List[str]) -> int:
    allowedDict = {}
    for i in allowed:
        allowedDict[i] = True
    count = 0
    for word in words:
        isValid = True
        for char in word:
            if char not in allowedDict:
                isValid = False
        if isValid:
            count += 1
    return count


# https://leetcode.com/problems/fibonacci-number/description/
def fib(self, n: int) -> int:
    results = [0,1]
    for i in range(2, n+1):
        results.append(results[i-1]+results[i-2])
    return results[n]

# https://leetcode.com/problems/get-maximum-in-generated-array/description/
def getMaximumGenerated(self, n: int) -> int:
    maxNums = [0,1]
    if (n <= 1):
        return maxNums[n]
    currMax = 1
    for i in range (2,n+1):
        temp = maxNums[i//2]
        if i % 2 != 0:
            temp += maxNums[i//2+1]
        if temp > currMax:
            currMax = temp
        maxNums.append(temp)
    return currMax

# https://leetcode.com/problems/find-maximum-number-of-string-pairs/description/
def maximumNumberOfStringPairs(self, words: List[str]) -> int:
    usedIndexDict = {}
    for i in range (0, len(words)):
        if i in usedIndexDict:
            continue
        for j in range (i+1, len(words)):
            if words[i] == words[j][::-1]:
                usedIndexDict[i] = True
                usedIndexDict[j] = True
    return len(usedIndexDict)//2


# https://leetcode.com/problems/rings-and-rods/description/
def countPoints(self, rings: str) -> int:
    rodDict = {}
    for i in range(0, 10):
        rodDict[str(i)] = {}
    
    for i in range(0, len(rings), 2):
        if rings[i] not in rodDict[rings[i+1]]:
            rodDict[rings[i+1]][rings[i]] = True
    count = 0
    for i in rodDict.keys():
        if len(rodDict[i]) == 3:
            count += 1
    return count

# https://leetcode.com/problems/sort-the-people/description/
def sortPeople(self, names: List[str], heights: List[int]) -> List[str]:
    heightToNameDict = {}
    for i in range (0, len(heights)):
        heightToNameDict[heights[i]] = names[i]
    
    sortedNames = []
    sortedHeights = sorted(heights, reverse=True)
    
    for i in sortedHeights:
        sortedNames.append(heightToNameDict[i])
    return sortedNames

# https://leetcode.com/problems/find-a-corresponding-node-of-a-binary-tree-in-a-clone-of-that-tree/description/
def getTargetCopy(self, original: TreeNode, cloned: TreeNode, target: TreeNode) -> TreeNode:
    foundLeftNode = None
    foundRightNode = None
    if cloned.val == target.val:
        return cloned
    if cloned.left == None and cloned.right == None:
        return None
    if cloned.left != None:
        foundLeftNode = self.getTargetCopy(original, cloned.left, target)
    if cloned.right != None:
        foundRightNode = self.getTargetCopy(original, cloned.right, target)
    if foundLeftNode != None:
        return foundLeftNode
    else:
        return foundRightNode

# https://leetcode.com/problems/find-the-k-beauty-of-a-number/description/    
def divisorSubstrings(self, num: int, k: int) -> int:
    kBeauty = 0
    for i in range(0, len(str(num))-k+1):
        if int(str(num)[i:i+k]) > 0 and num % int(str(num)[i:i+k]) == 0:
            kBeauty += 1
    return kBeauty

# https://leetcode.com/problems/minimum-recolors-to-get-k-consecutive-black-blocks/description/
def minimumRecolors(self, blocks: str, k: int) -> int:
    minRecolor = 100
    for i in range(0, len(blocks)-k+1):
        currentRecolor = self.countWhites(blocks[i:i+k])
        if currentRecolor < minRecolor:
            minRecolor = currentRecolor
        if minRecolor == 0:
            return minRecolor
    return minRecolor

def countWhites(self, subBlock: str):
    numOfWhites = 0
    for i in subBlock:
        if i == 'W':
            numOfWhites += 1
    return numOfWhites


# https://leetcode.com/problems/minimum-difference-between-highest-and-lowest-of-k-scores/
def minimumDifference(self, nums: List[int], k: int) -> int:
        sortedNums = sorted(nums, reverse=True)
        minDiff = -1
        for i in range(0, len(sortedNums)-(k-1)):
            diff = sortedNums[i]-sortedNums[i+]
            if minDiff == -1 or diff < minDiff:
                minDiff = diff
        if minDiff == -1:
            return 0
        return minDiff

# https://leetcode.com/problems/group-the-people-given-the-group-size-they-belong-to/
def groupThePeople(self, groupSizes: List[int]) -> List[List[int]]:
    groupSizeDict = {}
    groups = []
    for i in range(len(groupSizes)):
        groupSize = groupSizes[i]
        if groupSize not in groupSizeDict:
            groupSizeDict[groupSize] = [i]
        else:
            groupSizeDict[groupSize].append(i)
        
        if len(groupSizeDict[groupSize]) == groupSize:
            groups.append(groupSizeDict[groupSize])
            groupSizeDict.pop(groupSize)
    return group

# https://leetcode.com/problems/convert-an-array-into-a-2d-array-with-conditions/
def findMatrix(self, nums: List[int]) -> List[List[int]]:
    numDict = self.countOccurrences(nums)
    return self.build2DArrays(numDict, len(nums))

def countOccurrences(self, nums: List[int]) -> Dict[int, int]:
    solutions = {}
    for i in nums:
        if i in solutions:
            solutions[i] += 1
        else:
            solutions[i] = 1
    return solutions

def build2DArrays(self, numDict: Dict[int, int], totalNum) -> List[List[int]]:
    solutions = []
    while totalNum > 0:
        currentRow = []
        for i in numDict.keys():
            if numDict[i] > 0:
                currentRow.append(i)
                numDict[i] -= 1
                totalNum -= 1
        solutions.append(currentRow)

    return solutions

# https://leetcode.com/problems/arithmetic-subarrays/
def checkArithmeticSubarrays(self, nums: List[int], l: List[int], r: List[int]) -> List[bool]:
    solutions = []

    for i in range(len(l)):
        solutions.append(self.sortAndCheckSequence(nums[l[i]:r[i]+1]))
    return solutions

def sortAndCheckSequence(self, seqArray: List[int]) -> bool:
    seqArray.sort()
    diff = -1
    for i in range(0, len(seqArray)-1):
        if diff == -1:
            diff = seqArray[i+1] - seqArray[i]
        elif seqArray[i+1] - seqArray[i] != diff:
            return False
    return True

# https://leetcode.com/problems/flood-fill/
def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
    visitedCells = self.buildVisitedCells(image)
    self.floodFillRec(image, visitedCells, sr, sc, image[sr][sc], color)
    return image

def buildVisitedCells(self, image: List[List[int]]) -> List[List[int]]:
    visitedCells = []
    for i in range(len(image)):
        row = []
        for j in range(len(image[i])):
            row.append(False)
        visitedCells.append(row)
    return visitedCells

def floodFillRec(self, image: List[List[int]], visited: List[List[int]], sr: int, sc: int, startingColor: int, color: int) -> List[List[int]]:
    image[sr][sc] = color
    visited[sr][sc] = True
    # up
    if sr-1 >= 0 and not visited[sr-1][sc] and image[sr-1][sc] == startingColor:
        self.floodFillRec(image, visited, sr-1, sc, startingColor, color)
    
    # down
    if sr+1 < len(image) and not visited[sr+1][sc] and image[sr+1][sc] == startingColor:
        self.floodFillRec(image, visited, sr+1, sc, startingColor, color)

    # left
    if sc-1 >= 0 and not visited[sr][sc-1] and image[sr][sc-1] == startingColor:
        self.floodFillRec(image, visited, sr, sc-1, startingColor, color)
    
    # right
    if sc+1 < len(image[sr]) and not visited[sr][sc+1] and image[sr][sc+1] == startingColor:
        self.floodFillRec(image, visited, sr, sc+1, startingColor, color)


# https://leetcode.com/problems/maximum-depth-of-n-ary-tree/
def maxDepth(self, root: 'Node') -> int:
    if root == None:
        return 0
    return self.maxDepthRec(root, 0)

def maxDepthRec(self, node: 'Node', currHeight: int) -> int:
    currLevel = currHeight

    for i in node.children:
        childHeight = self.maxDepthRec(i, currHeight)
        if childHeight > currLevel:
            currLevel = childHeight

    return currLevel + 1

# https://leetcode.com/problems/binary-tree-level-order-traversal/
def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
        return []
    nodes = deque()
    nodes.append((root,0))
    levels = []

    while nodes:
        node, depth = nodes.popleft()
        if len(levels) == depth:
            levels.append([])
        levels[depth].append(node.val)
        if node.left:
            nodes.append((node.left, depth+1))
        if node.right:
            nodes.append((node.right, depth+1))
    return levels

# https://leetcode.com/problems/deepest-leaves-sum/
def deepestLeavesSum(self, root: Optional[TreeNode]) -> int:
    if not root:
        return 0
    currLevel = 0
    currSum = 0
    nodes = deque()
    nodes.append((1, root))
    while nodes:
        level, node = nodes.popleft()
        if currLevel < level:
            currLevel = level
            currSum = node.val
        else:
            currSum += node.val
        if node.left:
            nodes.append((level+1, node.left))
        if node.right:
            nodes.append((level+1, node.right))
    return currSum

# https://leetcode.com/problems/longest-nice-substring/
def longestNiceSubstring(self, s: str) -> str:

    if len(s)<2:
        return ""
    uniqueChar = set(s)
    pivotIndex = self.findPivotIndex(s, uniqueChar)
    if pivotIndex == -1:
        return s
    leftSubstring = self.longestNiceSubstring(s[:pivotIndex])
    rightSubstring = self.longestNiceSubstring(s[pivotIndex+1:])
    if len(rightSubstring) > len(leftSubstring):
        return rightSubstring
    return leftSubstring
    

def findPivotIndex(self, s: str, uniqueChar: Set[str]) -> int:
    for i,e in enumerate(s):
        if e.swapcase() not in uniqueChar:
            return i
    return -1


# https://leetcode.com/problems/reverse-bits/
def reverseBits(self, n: int) -> int:
    print(n)
    binaryString = ""
    temp = n
    while temp > 0:
        binaryString += "{}".format(temp%2)
        temp//=2
    binaryString += "0"*(32-len(binaryString))
    
    power = 0
    result = 0
    for i in range(len(binaryString)-1, -1, -1):
        print(binaryString[i])
        result += (int(binaryString[i]) * 2**power)
        power += 1
    return result

def reverseBitsV2(self, n: int) -> int:
    print(n)
    binaryString = ""
    temp = n
    power = 31
    result = 0
    while temp > 0:
        result += (temp%2) * 2**power
        power -=1
        temp//=2
    return result

# https://leetcode.com/problems/min-cost-climbing-stairs/
def minCostClimbingStairs(self, cost: List[int]) -> int:
    totalCosts = [0]*len(cost)
    for i in range(len(cost)-1, -1, -1):
        if len(cost)-1 - i < 2:
            totalCosts[i] = cost[i]
        else:
            totalCosts[i] = cost[i] + min(totalCosts[i+1], totalCosts[i+2])
    return min(totalCosts[0], totalCosts[1])

# https://leetcode.com/problems/n-th-tribonacci-number/
def tribonacci(self, n: int) -> int:
    def tribonacciRec(n: int, temp: List[int]) -> int:
        if temp[n] > -1:
            return temp[n]
        temp[n] = tribonacciRec(n-1, temp) + tribonacciRec(n-2, temp) + tribonacciRec(n-3, temp)
        return temp[n]
    
    tempResult = [-1]*38
    tempResult[0] = 0
    tempResult[1] = 1
    tempResult[2] = 1
    return tribonacciRec(n, tempResult)

def tribonacciV2(self, n: int) -> int:
    tempResult = [-1]*38
    tempResult[0] = 0
    tempResult[1] = 1
    tempResult[2] = 1
    for i in range(3, n+1):
        tempResult[i] = tempResult[i-1] + tempResult[i-2] + tempResult[i-3]
    return tempResult[n]


# https://leetcode.com/problems/is-subsequence/
def isSubsequence(self, s: str, t: str) -> bool:
    if len(s) == 0:
        return True
    tempS = s
    i = 0
    while i < len(t) and len(tempS) > 0:
        if tempS[0] == t[i]:
            tempS = tempS[1:]
        i+=1
    
    return len(tempS) == 0

# dynamic programming
def isSubsequenceV2(self, s: str, t: str) -> bool:
    nextPointer = [{}] * (len(t) + 1)
    for i in range(len(t)-1, -1, -1):
        nextPointer[i] = nextPointer[i+1].copy()
        nextPointer[i][t[i]] = i+1
    print(nextPointer)
    pointer = 0
    for i in s:
        if i in nextPointer[pointer]:
            pointer = nextPointer[pointer][i]
        else:
            return False
    return True

# https://leetcode.com/problems/lexicographically-smallest-palindrome/
def makeSmallestPalindrome(self, s: str) -> str:
    left = len(s) // 2 - 1
    right = -1
    result = ""
    if len(s) % 2 == 0:
        right = left + 1
    else:
        result = s[left+1]
        right = left + 2
    print(result)
    while right < len(s) and left >= 0:
        if s[right] != s[left]:
            if ord(s[right]) < ord(s[left]):
                result = s[right] + result + s[right]
            else:
                result = s[left] + result + s[left]
        else:
            result = s[left] + result + s[left]
        left -= 1
        right += 1
    return result

# with charArray, better runtime
def makeSmallestPalindromeV2(self, s: str) -> str:
    left = len(s) // 2 - 1
    right = -1
    result = list(s)
    if len(s) % 2 == 0:
        right = left + 1
    else:
        right = left + 2
    while right < len(result) and left >= 0:
        if result[right] != result[left]:
            if ord(result[right]) < ord(result[left]):
                result[left] = result[right]
            else:
                result[right] = result[left]
        left -= 1
        right += 1
    return "".join(result)


# https://leetcode.com/problems/minimum-sum-of-four-digit-number-after-splitting-digits/
def minimumSum(self, num: int) -> int:
    def convertToSortedNumArray(num: int) -> List[int]:
        result = []
        while num > 0:
            result.append(num % 10)
            num //= 10
        return sorted(result)
    
    def sumNumArray(nums: List[num]) -> int:
        total = 0
        for i in range(0, len(nums)-2):
            total += nums[i] * 10 + nums[i+2]
        return total
    
    numArray = convertToSortedNumArray(num)
    return sumNumArray(numArray)

# https://leetcode.com/problems/longest-harmonious-subsequence/
def findLHS(self, nums: List[int]) -> int:
    occurrence = {}
    for i in nums:
        if i in occurrence:
            occurrence[i] += 1
        else:
            occurrence[i] = 1
    subseq = 0
    sortedKeys = sorted(occurrence.keys())
    for i in range(0, len(occurrence)-1):
        if abs(sortedKeys[i] - sortedKeys[i+1]) == 1:
            subseq = max(subseq, occurrence[sortedKeys[i]] + occurrence[sortedKeys[i+1]])
    return subseq


# https://leetcode.com/problems/binary-search-tree-to-greater-sum-tree/description/?envType=problem-list-v2&envId=tree
# 1038. Binary Search Tree to Greater Sum Tree
def minMovesToSeat(self, seats: List[int], students: List[int]) -> int:
    seats.sort()
    students.sort()
    moves = 0
    for i in range(0, len(seats)):
        moves += abs(seats[i]-students[i])
    return moves

class Solution:
    def bstToGst(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        [currTree, _] = self.bstToGstRec(root, 0)
        return currTree
    
    def bstToGstRec(self, root: Optional[TreeNode], maxVal: int) -> [Optional[TreeNode], int]:
        currMax = maxVal
        currTree = TreeNode()
        rightSubtree = None
        leftSubtree = None
        if root.right is not None:
            [rightSubtree, currGreater] = self.bstToGstRec(root.right, currMax)
            currMax = max(currMax, currGreater)
        val = currMax = root.val + currMax
        if root.left is not None:
            [leftSubtree, currGreater] =  self.bstToGstRec(root.left, currMax)
            currMax = max(currGreater, currMax)
            
        return TreeNode(val=val, left=leftSubtree, right=rightSubtree), currMax

# https://leetcode.com/problems/subrectangle-queries/?envType=problem-list-v2&envId=matrix
# 1476. Subrectangle Queries 
class SubrectangleQueries:
    rectangle = None

    def __init__(self, rectangle: List[List[int]]):
        self.rectangle = rectangle

    def updateSubrectangle(self, row1: int, col1: int, row2: int, col2: int, newValue: int) -> None:
        for i in range(row1, row2+1):
            for j in range(col1, col2+1):
                self.rectangle[i][j] = newValue

    def getValue(self, row: int, col: int) -> int:
        return self.rectangle[row][col]

# https://leetcode.com/problems/sort-the-students-by-their-kth-score/?envType=problem-list-v2&envId=matrix
# 2545. Sort the Students by Their Kth Score
class Solution:
    def sortTheStudents(self, score: List[List[int]], k: int) -> List[List[int]]:
        examToStudentDict = {}
        scoreList = []
        for i in range(0, len(score)):
            examToStudentDict[score[i][k]] = i
            scoreList.append(score[i][k])
        scoreList.sort()
        sortedScoreList = []
        for i in range(len(scoreList)-1, -1, -1):
            sortedScoreList.append(score[examToStudentDict[scoreList[i]]])
        return sortedScoreList
# https://leetcode.com/problems/construct-smallest-number-from-di-string/description/?envType=problem-list-v2&envId=greedy
# 2375. Construct Smallest Number From DI String
class Solution:
    def smallestNumber(self, pattern: str) -> str:
        indexOfIFollowedByDToNumOfDDict = {}
        i = 0
        prevD = None
        while i < len(pattern):
            if pattern[i] == 'D':
                if prevD is None:
                    indexOfIFollowedByDToNumOfDDict[i-1] = 1
                    prevD = i-1
                else:
                    indexOfIFollowedByDToNumOfDDict[prevD] += 1
            else:
                prevD = None
            i += 1
        
        usedNumDict = {}
        strNum = ""
        currNum = 1

        # handle first character
        if -1 in indexOfIFollowedByDToNumOfDDict :
            currNum += indexOfIFollowedByDToNumOfDDict[-1]
        if pattern[0] == 'I':
            currNum = self.findUnusedHigherNum(currNum, usedNumDict)
        else:
            currNum = self.findUnusedLowerNum(currNum, usedNumDict)
        strNum += str(currNum)
        usedNumDict[currNum] = True
        maxNum = currNum
        
        
        for i in range(0, len(pattern)):
            if i in indexOfIFollowedByDToNumOfDDict:
                currNum = maxNum + indexOfIFollowedByDToNumOfDDict[i]+1
            if pattern[i] == 'I':
                currNum = self.findUnusedHigherNum(currNum, usedNumDict)
            else:
                currNum = self.findUnusedLowerNum(currNum, usedNumDict)
            usedNumDict[currNum] = True
            strNum += str(currNum)
            maxNum = max(maxNum, currNum)
        return strNum
    
    def findUnusedHigherNum(self, currNum: int, usedNumDict: dict) -> int:
        for i in range(currNum, 10):
            if i not in usedNumDict:
                return i
        return -1
            
    def findUnusedLowerNum(self, currNum: int, usedNumDict: dict) -> int:
        for i in range(currNum, 0, -1):
            if i not in usedNumDict:
                return i
        return -1
    
# https://leetcode.com/problems/maximum-absolute-sum-of-any-subarray/
# 1749. Maximum Absolute Sum of Any Subarray
class Solution:
    def maxAbsoluteSum(self, nums: List[int]) -> int:
        # using kadane algorithm
        # if maxEndingAtI is negative, always better to reset the maxCurrentIndex
        # if maxEndingAtI is positive always better to extend

        maxMax = nums[0]
        maxEndingAtI = nums[0]

        for i in range(1, len(nums)):
            maxEndingAtI = max(maxEndingAtI + nums[i], nums[i])
            maxMax = max(maxEndingAtI, maxMax)

        # same as previous kadane but find minSubArray
        
        minMin = nums[0]
        minEndingAtI = nums[0]
        for i in range(1, len(nums)):
            minEndingAtI = min(minEndingAtI + nums[i], nums[i])
            minMin = min(minEndingAtI, minMin)
        return max(maxMax, abs(minMin))
    
# https://leetcode.com/problems/house-robber/description/?envType=study-plan-v2&envId=dynamic-programming
# 198. House Robber
class Solution:
    def rob(self, nums: List[int]) -> int:
        maxRobAtIndexI = [0] * len(nums)
        currentMax = 0
        for i in range(0, len(nums)):
            prev2 = maxRobAtIndexI[i-2] if (i-2 >= 0) else 0
            prev1 = maxRobAtIndexI[i-1] if (i-1 >= 0) else 0
            maxRobAtIndexI[i] = max(prev2+nums[i], prev1)
            currentMax = max(maxRobAtIndexI[i], currentMax)
        return currentMax
    

# https://leetcode.com/problems/apply-operations-to-an-array/description/?envType=daily-question&envId=2025-03-01
# 2025. Apply Operations to an Array
    
class Solution:
    def applyOperations(self, nums: List[int]) -> List[int]:
        i = 0
        while i < len(nums):
            if i+1 < len(nums) and nums[i] == nums[i+1]:
                nums[i] *= 2
                nums[i+1] = 0
            i += 1
        
        i = 0
        j = 1
        while i < len(nums) and j < len(nums):
            if nums[i] == 0:
                if nums[j] != 0:
                    nums[i], nums[j] = nums[j], nums[i]
                    i += 1
                    j += 1
                else:
                    j += 1
            else:
                i += 1
                j += 1
        return nums

# https://leetcode.com/problems/longest-palindromic-substring/    
# 5. Longest Palindromic Substring
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if len(s) == 1:
            return s

        lengthToStartIndexDict = {}
        maxSubstring = s[0]
        for i in range(0, len(s)):
            if 1 in lengthToStartIndexDict:
                lengthToStartIndexDict[1].append(i)
            else:
                lengthToStartIndexDict[1] = [i]
        
        for i in range(0, len(s)):
            if i+1 < len(s) and s[i] == s[i+1]:
                if 2 in lengthToStartIndexDict:
                    lengthToStartIndexDict[2].append(i)
                else:
                    lengthToStartIndexDict[2] = [i]
                    if len(maxSubstring) < 2:
                        maxSubstring = s[i:i+2]
        
        for i in range(3, len(s)+1):
            palindromeList = []
            prevPalindrome = lengthToStartIndexDict[i-2] if i-2 in lengthToStartIndexDict else []
            for j in range(0, len(prevPalindrome)):
                startIndex = prevPalindrome[j]
                if startIndex - 1 >= 0 and startIndex + i - 2 < len(s) and s[startIndex - 1] == s[startIndex + i - 2]:
                    palindromeList.append(startIndex - 1)
                    if len(s[startIndex-1:startIndex + i - 1]) > len(maxSubstring):
                        maxSubstring = s[startIndex-1:startIndex + i - 1]
            lengthToStartIndexDict[i] = palindromeList
        return maxSubstring
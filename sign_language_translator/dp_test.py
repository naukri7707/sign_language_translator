# DP 問題:

# 兩個陣列 a, b

# a, b 中有數量不固定的點 (x,y)

# 點和點的偏移使用曼哈頓距離計算

# a,b 中的點只能是一對一的關係 e.g a1 已經與 b1 產生聯繫，則 a1 不能與 b2 產生聯繫
# 最佳匹配的情況是所有匹配相加的距離盡可能小

# 如果 len(a) > len(b) 需要為每個 b 點找到最佳匹配的 a 點            
# 如果 len(b) > len(a) 需要為每個 a 點找到最佳匹配的 b 點        

# 你最後需要輸出一個陣列，該陣列是陣列 b 中與 a 點最佳匹的點的索引值 ，如果
# len(a) > len(b) 則將沒有匹配到的點輸出 -1

# e.g 

# a[(0,0), (0,1), (0,2)]
# b[(0,0), (0,2)]

# 則輸出: [0, -1, 1]

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def min_distance_matching(a, b):
    len_a = len(a)
    len_b = len(b)

    # 創建曼哈頓距離矩陣
    distances = [[0] * len_b for _ in range(len_a)]
    for i in range(len_a):
        for j in range(len_b):
            distances[i][j] = manhattan_distance(a[i], b[j])

    # 使用 DP 找到距離總和最短的組合
    for i in range(1, len_a):
        distances[i][0] += distances[i - 1][0]

    return result

# 測試
a = [(0, 0), (0, 1), (0, 2), (0, 3)]
b = [(0, 0), (0, 2), (0, 4)]
result = min_distance_matching(a, b)
print(result)

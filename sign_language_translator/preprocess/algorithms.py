import sys
from typing import Any, List, Callable, TypeVar

T = TypeVar('T')
TDistance = TypeVar('TDistance')

def manhattan_distance(p1: T, p2: T) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def create_distance_map(
        row: List[T],
        col: List[T],
        get_distance: Callable[[T, T], TDistance] = manhattan_distance
        ) -> List[List[TDistance]]:

    row = row if row else []
    col = col if col else []

    len_origin = len(row)
    len_target = len(col)

    # 建立曼哈頓距離表
    map = [[0] * len_target for _ in range(len_origin)]
    for i in range(len_origin):
        for j in range(len_target):
            map[i][j] = get_distance(row[i], col[j])
    
    return map

def calc_min_distance_matching(map: List[List[TDistance]]) -> (TDistance, List[TDistance]):
    row_count = len(map)
    column_count = 0 if row_count == 0 else len(map[0])
    match_count = min(row_count, column_count)
    
    best_distance = sys.maxsize
    best_matches = []
    match = []
    match_point_count = 0

    def dfs(dis, i):
        nonlocal best_distance, best_matches, match_point_count
        
        # 剪枝 : 如果當前分數已經大於最佳分數，則不再繼續
        # if dis > best_distance:
        #     return

        # 如果已經找到所有點的匹配，則嘗試更新最佳分數
        if match_point_count == match_count:
            if(dis < best_distance):
                best_distance = dis
                best_matches = match.copy()
            return
        
        # 如果已經遍歷完所有 row 點，則不再繼續
        if i == row_count:
            return

        for j in range(-1, column_count):
            # 如果當前 b 點已經被匹配，則不再繼續
            if j != -1 and j in match:
                continue

            # 
            match.append(j)
            match_point_count += 1 if j != -1 else 0
            addition_score = 0 if j == -1 else map[i][j]
            
            #
            dfs(dis + addition_score, i + 1)
            
            #
            match.pop()
            match_point_count -= 1 if j != -1 else 0
        pass
    
    dfs(0,0)

    return (best_distance, best_matches)
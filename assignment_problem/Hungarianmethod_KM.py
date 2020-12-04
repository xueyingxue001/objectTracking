#-*-coding:utf-8-*-
'''
7 条边
0 0
0 1
1 1
1 2
2 0
2 1
3 2
'''
line = [[0.8, 0.6, 0.0, 0.0],  # 1
        [0.0, 0.3, 0.9, 0.0],  # 2
        [0.9, 0.8, 0.0, 0.0],  # 3
        [0.0, 0.0, 0.2, 0.0]]  # 4

# girls[i] = 匹配到 ? 1 : -1
# girl_with_boy[i] = j: j 号男生匹配给 i 号女生 
girl_with_boy, girls_check = [-1.0] * len(line), [0] * len(line[0])

# Work for KM
boys, girls = [0.0] * len(line), [0.0] * len(line[0])
for i in range(len(line)): boys[i] = max(line[i])

def KMFind(cur_boy_id):
    global girls_check, girl_with_boy, boys, girls
    d = 0.1
    for i in range(len(girls_check)):
        if (not line[cur_boy_id][i] or girls_check[i]): continue
        # 如果 line 的值 > boy and girl 得期望值才继续
        if (line[cur_boy_id][i] < boys[cur_boy_id] + girls[i]): continue

        girls_check[i] = 1

        if (girl_with_boy[i] == -1 or KMFind(girl_with_boy[i])):
            girl_with_boy[i] = cur_boy_id
            return True
        # boy 匹配或更换失败一次，期望值 -1(天下的芳草与我无关)
        # girl 看到对方除了自己外不选别人，则期望值 +1
        boys[cur_boy_id] -= d
        boys[girl_with_boy[i]] -= d
        girls[i] += d

        if (boys[cur_boy_id] < 0): 
            return False
    return False

def Find(cur_boy_id):
    global girls_check, girl_with_boy
    # 遍历所有女生，看哪个女生愿意匹配该男生
    for i in range(len(girls_check)):
        # 若发生如下情况的任何一种，则不将 girls[i] 匹配给 cur_boy_id
        #       1，cur_boy_id 与 girls[i] 互相无意向(不能乱牵线)
        #       2，girls[i] 已经被考察过
        if (not line[cur_boy_id][i] or girls_check[i]): continue
        
        # 将 girls[i] 标记为 "已考察"，并开始考察
        girls_check[i] = 1

        # 若当前 girl 还单身(girl_with_boy[i] == -1)，则将 cur_boy_id 匹配给当前的 girl
        # 若当前 girl 已配给男生 j，则看看能否把男生 j 配给其他女生
        #      能得话就男生 j 配给其他女生，将 girl[i] 配给 cur_boy_id
        if (girl_with_boy[i] == -1 or Find(girl_with_boy[i])):
            girl_with_boy[i] = cur_boy_id # 将 cur_boy_id 号男生匹配给 i 号女生
            return True
    return False

def Match(type_name="Hungarian"):
    global girls_check, girl_with_boy
    girl_with_boy, girls_check = [-1.0] * len(line), [0] * len(line[0])

    sum = 0
    for i in range(len(girl_with_boy)): # 遍历每个男生，看其能否被匹配到
        girls_check = [0] * len(girls_check)
        # 如果当前的 girl_with_boy 有一个 girls 可以于其匹配，则 sum++
        if (type_name == "Hungarian"):
            if Find(i): sum += 1
        else:
            if KMFind(i): sum += 1
    print("\n-----------------\n{} couples:".format(sum))
    for i in range(len(girl_with_boy)):
        if girl_with_boy[i] != -1:
            print("  Boy {} ---- Girl {}".format(girl_with_boy[i], i))
            
    return sum

if __name__ == '__main__':
    Match()
    Match("KM")

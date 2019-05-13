a = [['GH신소재', '3,645', '4,115', '3,935', '-801,819', '-5.35', '130500']]
b = [['GH신소재', '3,645', '4,115', '3,940', '-783,708', '-5.23', '130500']]

def list_diff(li1, li2):
    s_list1 = []
    s_list2 = []
    for stock1 in li1:
        s_list1.append(stock1[6])
    for stock2 in li2:
        s_list2.append(stock2[6])
    li3 = (list(set(s_list1) - set(s_list2)))
    li4 = (list(set(s_list2) - set(s_list1)))
    if len(li3 + li4) > 0:
        return True
    else:
        return False

print(list_diff(a,b))
import json
from json import JSONDecoder, JSONDecodeError
import re
import ast
import numpy as np

if __name__ == '__main__':
    # all_review_data = open("restaurants.json", encoding="utf-8")
    # for line in all_review_data:
    #     curr_dict = ast.literal_eval(line)
    #     print(curr_dict["business_id"])
    list1 = np.ones(1000)
    list2 = np.zeros(1000)
    for i, (j, k) in enumerate(zip(list1, list2)):
        print(i)
        print(j)
        print(k)

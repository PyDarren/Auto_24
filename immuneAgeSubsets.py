# Title     : TODO
# Objective : TODO
# Created by: Chen Da
# Created on: 2020/1/14



import pandas as pd
import numpy as np
import os


class SubsetsRatio():

    def __init__(self, raw_df):
        self.raw_df = raw_df


    def marker1_num(self, marker, label):
        select_df = self.raw_df[self.raw_df[marker] == label]
        return len(select_df), select_df


    def marker2_num(self, marker1, marker2, label1, label2):
        '''
        计算2个marker确定亚群的细胞数
        :param marker1: 第一个marker的名字
        :param marker2: 第二个marker的名字
        :param label1: 第一个marker的类别
        :param label2: 第二个marker的类别
        :return: 该2个marker所确定的细胞数
        '''
        marker1_length = len(self.raw_df[self.raw_df[marker1] == label1])
        marker2_length = len(self.raw_df[self.raw_df[marker2] == label2])
        if marker1_length >= marker2_length:
            select_df = self.raw_df[self.raw_df[marker1] == label1]
            select_df = select_df[select_df[marker2] == label2]
        else:
            select_df = self.raw_df[self.raw_df[marker2] == label2]
            select_df = select_df[select_df[marker1] == label1]
        return len(select_df), select_df


    def marker3_num(self, marker1, marker2, marker3, label1, label2, label3):
        '''
        计算3个marker确定亚群的细胞数
        :param marker1: 第一个marker的名字
        :param marker2: 第二个marker的名字
        :param marker3: 第三个marker的名字
        :param label1: 第一个marker的类别
        :param label2: 第二个marker的类别
        :param label3: 第三个marker的类别
        :return: 该3个marker所确定的细胞数
        '''
        marker1_length = len(self.raw_df[self.raw_df[marker1] == label1])
        marker2_length = len(self.raw_df[self.raw_df[marker2] == label2])
        if marker1_length >= marker2_length:
            select_df = self.raw_df[self.raw_df[marker1] == label1]
            select_df = select_df[select_df[marker2] == label2]
        else:
            select_df = self.raw_df[self.raw_df[marker2] == label2]
            select_df = select_df[select_df[marker1] == label1]
        select_df = select_df[select_df[marker3] == label3]
        return len(select_df), select_df


    def marker4_num(self, marker1, marker2, marker3, marker4, label1, label2, label3, label4):
        '''
        计算3个marker确定亚群的细胞数
        :param marker1: 第一个marker的名字
        :param marker2: 第二个marker的名字
        :param marker3: 第三个marker的名字
        :param marker4: 第四个marker的名字
        :param label1: 第一个marker的类别
        :param label2: 第二个marker的类别
        :param label3: 第三个marker的类别
        :param label4: 第四个marker的类别
        :return: 该3个marker所确定的细胞数
        '''
        marker1_length = len(self.raw_df[self.raw_df[marker1] == label1])
        marker2_length = len(self.raw_df[self.raw_df[marker2] == label2])
        if marker1_length >= marker2_length:
            select_df = self.raw_df[self.raw_df[marker1] == label1]
            select_df = select_df[select_df[marker2] == label2]
        else:
            select_df = self.raw_df[self.raw_df[marker2] == label2]
            select_df = select_df[select_df[marker1] == label1]
        select_df = select_df[select_df[marker3] == label3]
        select_df = select_df[select_df[marker4] == label4]
        return len(select_df), select_df


    def marker5_num(self, marker1, marker2, marker3, marker4, marker5, label1, label2, label3, label4, label5):
        '''
        计算3个marker确定亚群的细胞数
        :param marker1: 第一个marker的名字
        :param marker2: 第二个marker的名字
        :param marker3: 第三个marker的名字
        :param marker4: 第四个marker的名字
        :param marker5: 第五个marker的名字
        :param label1: 第一个marker的类别
        :param label2: 第二个marker的类别
        :param label3: 第三个marker的类别
        :param label4: 第四个marker的类别
        :param marker5: 第五个marker的类别
        :return: 该3个marker所确定的细胞数
        '''
        marker1_length = len(self.raw_df[self.raw_df[marker1] == label1])
        marker2_length = len(self.raw_df[self.raw_df[marker2] == label2])
        if marker1_length >= marker2_length:
            select_df = self.raw_df[self.raw_df[marker1] == label1]
            select_df = select_df[select_df[marker2] == label2]
        else:
            select_df = self.raw_df[self.raw_df[marker2] == label2]
            select_df = select_df[select_df[marker1] == label1]
        select_df = select_df[select_df[marker3] == label3]
        select_df = select_df[select_df[marker4] == label4]
        select_df = select_df[select_df[marker5] == label5]
        return len(select_df), select_df


    def non_naive_cells(self, marker1='CD45RA', marker2='CD197', label1=0, label2=0):
        '''
        计算non naive cells个数
        :param marker1:
        :param marker2:
        :param label1:
        :param label2:
        :return:
        '''
        df = self.marker4_num('CD33', 'CD14', 'CD3', 'CD4', 1, 1, 0, 0)[1]
        marker1_length = len(df[df[marker1] == label1])
        marker2_length = len(df[df[marker2] == label2])
        if marker1_length >= marker2_length:
            select_df = df[df[marker1] == label1]
            select_df = select_df[select_df[marker2] == label2]
        else:
            select_df = df[df[marker2] == label2]
            select_df = select_df[select_df[marker1] == label1]
        non_naive_index = list(set(df.index).difference(select_df.index))
        non_naive = df.loc[non_naive_index, :]
        return len(non_naive), non_naive


    def cross_classification(self, marker1, marker2):
        '''
        计算十字分类后四个类别的个数
        :return:
        '''
        pos_and_pos = self.raw_df[np.logical_and(self.raw_df[marker1]==0, self.raw_df[marker2]==0)].shape[0]
        neg_and_neg = self.raw_df[np.logical_and(self.raw_df[marker1]==1, self.raw_df[marker2]==1)].shape[0]
        pos_and_neg = self.raw_df[np.logical_and(self.raw_df[marker1]==0, self.raw_df[marker2]==1)].shape[0]
        neg_and_pos = self.raw_df[np.logical_and(self.raw_df[marker1]==1, self.raw_df[marker2]==0)].shape[0]
        return pos_and_pos, neg_and_neg, pos_and_neg, neg_and_pos



def difference_set(df_more, df_less):
    '''
    计算两个数据框的差集
    :param df1:
    :param df2:
    :return:
    '''
    new_index = list(set(df_more.index).difference(df_less.index))
    new_df = df_more.loc[new_index, :]
    return new_df


def subsetsRatioCalculation_real(df):
    ratio_list =list()
    subset_ratio = SubsetsRatio(df)

    # 3. lymphocytes
    lymphocytes_num = subset_ratio.marker2_num('CD33', 'CD14', 1, 1)[0]
    lymphocytes_ratio = lymphocytes_num / len(df)
    print('亚群lymphocytes的比率为', lymphocytes_ratio)
    print('-'*100)
    ratio_list.append(lymphocytes_ratio)

    # 4. Lymphocytes/CD3+
    lymphocytes_CD3Pos_num = subset_ratio.marker3_num('CD33', 'CD14', 'CD3', 1, 1, 0)[0]
    lymphocytes_CD3Pos_ratio = lymphocytes_CD3Pos_num / lymphocytes_num
    print('亚群Lymphocytes/CD3+的比率为', lymphocytes_CD3Pos_ratio)
    print('-'*100)
    ratio_list.append(lymphocytes_CD3Pos_ratio)

    # 5. Lymphocytes/CD3+/CD4+
    lymphocytes_CD3Pos_CD4Pos_num = subset_ratio.marker4_num('CD33', 'CD14', 'CD3', 'CD4', 1, 1, 0, 0)[0]
    lymphocytes_CD3Pos_CD4Pos_ratio = lymphocytes_CD3Pos_CD4Pos_num / lymphocytes_CD3Pos_num
    print('亚群Lymphocytes/CD3+/CD4+的比率为', lymphocytes_CD3Pos_CD4Pos_ratio)
    print('-'*100)
    ratio_list.append(lymphocytes_CD3Pos_CD4Pos_ratio)

    # 10. Lymphocytes/CD3+/CD4+/CD57+
    lymphocytes_CD3Pos_CD4Pos_CD57Pos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD4', 'CD57', 1, 1, 0, 0, 0)[0]
    lymphocytes_CD3Pos_CD4Pos_CD57Pos_ratio = lymphocytes_CD3Pos_CD4Pos_CD57Pos_num / lymphocytes_CD3Pos_CD4Pos_num
    print('亚群Lymphocytes/CD3+/CD4+/CD57+的比率为', lymphocytes_CD3Pos_CD4Pos_CD57Pos_ratio)
    print('-'*100)
    ratio_list.append(lymphocytes_CD3Pos_CD4Pos_CD57Pos_ratio)

    # 13. Lymphocytes/CD3+/CD4+/CD94+
    lymphocytes_CD3Pos_CD4Pos_CD94Pos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD4', 'CD94', 1, 1, 0, 0, 0)[0]
    lymphocytes_CD3Pos_CD4Pos_CD94Pos_ratio = lymphocytes_CD3Pos_CD4Pos_CD94Pos_num / lymphocytes_CD3Pos_CD4Pos_num
    print('亚群Lymphocytes/CD3+/CD4+/CD94+的比率为', lymphocytes_CD3Pos_CD4Pos_CD94Pos_ratio)
    print('-'*100)
    ratio_list.append(lymphocytes_CD3Pos_CD4Pos_CD94Pos_ratio)

    # 14. Lymphocytes/CD3+/CD4+/CD94-
    lymphocytes_CD3Pos_CD4Pos_CD94Neg_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD4', 'CD94', 1, 1, 0, 0, 1)[0]
    lymphocytes_CD3Pos_CD4Pos_CD94Neg_ratio = lymphocytes_CD3Pos_CD4Pos_CD94Neg_num / lymphocytes_CD3Pos_CD4Pos_num
    print('亚群Lymphocytes/CD3+/CD4+/CD94-的比率为', lymphocytes_CD3Pos_CD4Pos_CD94Neg_ratio)
    print('-'*100)
    ratio_list.append(lymphocytes_CD3Pos_CD4Pos_CD94Neg_ratio)

    # 19. Lymphocytes/CD3+/CD4+/CD278+
    lymphocytes_CD3Pos_CD4Pos_CD278Pos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD4', 'CD278', 1, 1, 0, 0, 0)[0]
    lymphocytes_CD3Pos_CD4Pos_CD278Pos_ratio = lymphocytes_CD3Pos_CD4Pos_CD278Pos_num / lymphocytes_CD3Pos_CD4Pos_num
    print('亚群Lymphocytes/CD3+/CD4+/CD278+的比率为', lymphocytes_CD3Pos_CD4Pos_CD278Pos_ratio)
    print('-'*100)
    ratio_list.append(lymphocytes_CD3Pos_CD4Pos_CD278Pos_ratio)

    # 20. Lymphocytes/CD3+/CD4+/non-naive cells
    lymphocytes_CD3Pos_CD4Pos_non_naive_num = subset_ratio.non_naive_cells()[0]
    lymphocytes_CD3Pos_CD4Pos_non_naive_ratio = lymphocytes_CD3Pos_CD4Pos_non_naive_num / lymphocytes_CD3Pos_CD4Pos_num
    print('亚群Lymphocytes/CD3+/CD4+/non-naive cells的比率为', lymphocytes_CD3Pos_CD4Pos_non_naive_ratio)
    print('-'*100)
    ratio_list.append(lymphocytes_CD3Pos_CD4Pos_non_naive_ratio)

    # 21. Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5+
    non_naive_df = subset_ratio.non_naive_cells()[1]
    nonNaive = SubsetsRatio(non_naive_df)
    lymphocytes_CD3Pos_CD4Pos_non_naive_CXCR5Pos_num = nonNaive.marker1_num('CXCR5', 0)[0]
    lymphocytes_CD3Pos_CD4Pos_non_naive_CXCR5Pos_ratio = lymphocytes_CD3Pos_CD4Pos_non_naive_CXCR5Pos_num / lymphocytes_CD3Pos_CD4Pos_non_naive_num
    print('亚群Lymphocytes/CD3+/CD4+/non-naive/CXCR5+的比率为', lymphocytes_CD3Pos_CD4Pos_non_naive_CXCR5Pos_ratio)
    print('-'*100)
    ratio_list.append(lymphocytes_CD3Pos_CD4Pos_non_naive_CXCR5Pos_ratio)

    # 22. Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5-
    lymphocytes_CD3Pos_CD4Pos_non_naive_CXCR5Neg_num = nonNaive.marker1_num('CXCR5', 1)[0]
    lymphocytes_CD3Pos_CD4Pos_non_naive_CXCR5Neg_ratio = lymphocytes_CD3Pos_CD4Pos_non_naive_CXCR5Neg_num / lymphocytes_CD3Pos_CD4Pos_non_naive_num
    print('亚群Lymphocytes/CD3+/CD4+/non-naive/CXCR5-的比率为', lymphocytes_CD3Pos_CD4Pos_non_naive_CXCR5Neg_ratio)
    print('-'*100)
    ratio_list.append(lymphocytes_CD3Pos_CD4Pos_non_naive_CXCR5Neg_ratio)

    # 23. Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5-/CXCR3+CCR6+
    CXCR5Neg_df = nonNaive.marker1_num('CXCR5', 1)[1]
    CXCR5Neg = SubsetsRatio(CXCR5Neg_df)
    CXCR3_CCR6 = CXCR5Neg.cross_classification('CD183', 'CCR6')
    CXCR3Pos_CCR6Pos_num = CXCR3_CCR6[0]
    CXCR3Pos_CCR6Pos_ratio = CXCR3Pos_CCR6Pos_num / lymphocytes_CD3Pos_CD4Pos_non_naive_CXCR5Neg_num
    print('亚群Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5-/CXCR3+CCR6+的比率为', CXCR3Pos_CCR6Pos_ratio)
    print('-'*100)
    ratio_list.append(CXCR3Pos_CCR6Pos_ratio)

    # 24. Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5-/CXCR3+CCR6-
    CXCR3Pos_CCR6Neg_num = CXCR3_CCR6[2]
    CXCR3Pos_CCR6Neg_ratio = CXCR3Pos_CCR6Neg_num / lymphocytes_CD3Pos_CD4Pos_non_naive_CXCR5Neg_num
    print('亚群Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5-/CXCR3+CCR6-的比率为', CXCR3Pos_CCR6Neg_ratio)
    print('-'*100)
    ratio_list.append(CXCR3Pos_CCR6Neg_ratio)

    # 25. Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5-/CXCR3-CCR6-
    CXCR3Neg_CCR6Neg_num = CXCR3_CCR6[1]
    CXCR3Neg_CCR6Neg_ratio = CXCR3Neg_CCR6Neg_num / lymphocytes_CD3Pos_CD4Pos_non_naive_CXCR5Neg_num
    print('亚群Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5-/CXCR3-CCR6-的比率为', CXCR3Neg_CCR6Neg_ratio)
    print('-'*100)
    ratio_list.append(CXCR3Neg_CCR6Neg_ratio)

    # 26. Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5-/CXCR3-CCR6+
    CXCR3Neg_CCR6Pos_num = CXCR3_CCR6[3]
    CXCR3Neg_CCR6Pos_ratio = CXCR3Neg_CCR6Pos_num / lymphocytes_CD3Pos_CD4Pos_non_naive_CXCR5Neg_num
    print('亚群Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5-/CXCR3-CCR6+的比率为', CXCR3Neg_CCR6Pos_ratio)
    print('-'*100)
    ratio_list.append(CXCR3Neg_CCR6Pos_ratio)

    # 27. Lymphocytes/CD3+/CD4+/PD1+
    lymphocytes_CD3Pos_CD4Pos_PD1Pos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD4', 'PD1', 1, 1, 0, 0, 0)[0]
    lymphocytes_CD3Pos_CD4Pos_PD1Pos_ratio = lymphocytes_CD3Pos_CD4Pos_PD1Pos_num / lymphocytes_CD3Pos_CD4Pos_num
    print('亚群Lymphocytes/CD3+/CD4+/PD1+的比率为', lymphocytes_CD3Pos_CD4Pos_PD1Pos_ratio)
    print('-'*100)
    ratio_list.append(lymphocytes_CD3Pos_CD4Pos_PD1Pos_ratio)

    # 28. Lymphocytes/CD3+/CD4+/Q1: 158Gd_CD197_CCR7- , 155Gd_CD45RA+
    CD4Pos_df = subset_ratio.marker4_num('CD33', 'CD14', 'CD3', 'CD4', 1, 1, 0, 0)[1]
    CD4Pos = SubsetsRatio(CD4Pos_df)
    CD197_CD45RA = CD4Pos.cross_classification('CD197', 'CD45RA')
    CD197Neg_CD45RAPos_num = CD197_CD45RA[3]
    CD197Neg_CD45RAPos_ratio = CD197Neg_CD45RAPos_num / lymphocytes_CD3Pos_CD4Pos_num
    # CD197Neg_CD45RAPos_ratio *= 3
    print('亚群Lymphocytes/CD3+/CD4+/Q1: 158Gd_CD197_CCR7- , 155Gd_CD45RA+的比率为', CD197Neg_CD45RAPos_ratio)
    print('-'*100)
    ratio_list.append(CD197Neg_CD45RAPos_ratio)

    # 29. Lymphocytes/CD3+/CD4+/Q2: 158Gd_CD197_CCR7+ , 155Gd_CD45RA+
    CD197Pos_CD45RAPos_num = CD197_CD45RA[0]
    CD197Pos_CD45RAPos_ratio = CD197Pos_CD45RAPos_num / lymphocytes_CD3Pos_CD4Pos_num
    # CD197Pos_CD45RAPos_ratio *= 4
    print('亚群Lymphocytes/CD3+/CD4+/Q1: 158Gd_CD197_CCR7+ , 155Gd_CD45RA+的比率为', CD197Pos_CD45RAPos_ratio)
    print('-'*100)
    ratio_list.append(CD197Pos_CD45RAPos_ratio)

    # 30. Lymphocytes/CD3+/CD4+/Q3: 158Gd_CD197_CCR7+ , 155Gd_CD45RA-
    CD197Pos_CD45RANeg_num = CD197_CD45RA[2]
    CD197Pos_CD45RANeg_ratio = CD197Pos_CD45RANeg_num / lymphocytes_CD3Pos_CD4Pos_num
    print('亚群Lymphocytes/CD3+/CD4+/Q1: 158Gd_CD197_CCR7+ , 155Gd_CD45RA-的比率为', CD197Pos_CD45RANeg_ratio)
    print('-'*100)
    ratio_list.append(CD197Pos_CD45RANeg_ratio)

    # 31. Lymphocytes/CD3+/CD4+/Q4: 158Gd_CD197_CCR7- , 155Gd_CD45RA-
    CD197Neg_CD45RANeg_num = CD197_CD45RA[1]
    CD197Neg_CD45RANeg_ratio = CD197Neg_CD45RANeg_num / lymphocytes_CD3Pos_CD4Pos_num
    # CD197Neg_CD45RANeg_ratio /= 2
    print('亚群Lymphocytes/CD3+/CD4+/Q1: 158Gd_CD197_CCR7- , 155Gd_CD45RA-的比率为', CD197Neg_CD45RANeg_ratio)
    print('-'*100)
    ratio_list.append(CD197Neg_CD45RANeg_ratio)

    # 37. Lymphocytes/CD3+/CD4+/Tfh
    tfh_num = CD4Pos.marker2_num('CXCR5', 'PD1', 0, 0)[0]
    tfh_ratio = tfh_num / lymphocytes_CD3Pos_CD4Pos_num
    print('亚群Lymphocytes/CD3+/CD4+/Tfh的比率为', tfh_ratio)
    print('-'*100)
    ratio_list.append(tfh_ratio)

    # 38. Lymphocytes/CD3+/CD4+/Treg
    treg_num = CD4Pos.marker2_num('CD127', 'CD25', 1, 0)[0]
    treg_ratio = treg_num / lymphocytes_CD3Pos_CD4Pos_num
    print('亚群Lymphocytes/CD3+/CD4+/Treg的比率为', treg_ratio)
    print('-'*100)
    ratio_list.append(treg_ratio)

    # 43. Lymphocytes/CD3+/CD8+
    lymphocytes_CD3Pos_CD8Pos_num = subset_ratio.marker4_num('CD33', 'CD14', 'CD3', 'CD8', 1, 1, 0, 0)[0]
    lymphocytes_CD3Pos_CD8Pos_ratio = lymphocytes_CD3Pos_CD8Pos_num / lymphocytes_CD3Pos_num
    print('亚群Lymphocytes/CD3+/CD8+的比率为', lymphocytes_CD3Pos_CD8Pos_ratio)
    print('-'*100)
    ratio_list.append(lymphocytes_CD3Pos_CD8Pos_ratio)

    # 48. Lymphocytes/CD3+/CD8+/CD57+
    lymphocytes_CD3Pos_CD8Pos_CD57Pos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD8', 'CD57', 1, 1, 0, 0, 0)[0]
    lymphocytes_CD3Pos_CD8Pos_CD57Pos_ratio = lymphocytes_CD3Pos_CD8Pos_CD57Pos_num / lymphocytes_CD3Pos_CD8Pos_num
    print('亚群Lymphocytes/CD3+/CD8+/CD57+的比率为', lymphocytes_CD3Pos_CD8Pos_CD57Pos_ratio)
    print('-'*100)
    ratio_list.append(lymphocytes_CD3Pos_CD8Pos_CD57Pos_ratio)

    # 51. Lymphocytes/CD3+/CD8+/CD94+
    lymphocytes_CD3Pos_CD8Pos_CD94Pos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD8', 'CD94', 1, 1, 0, 0, 0)[0]
    lymphocytes_CD3Pos_CD8Pos_CD94Pos_ratio = lymphocytes_CD3Pos_CD8Pos_CD94Pos_num / lymphocytes_CD3Pos_CD8Pos_num
    print('亚群Lymphocytes/CD3+/CD8+/CD94+的比率为', lymphocytes_CD3Pos_CD8Pos_CD94Pos_ratio)
    print('-'*100)
    ratio_list.append(lymphocytes_CD3Pos_CD8Pos_CD94Pos_ratio)

    # 52. Lymphocytes/CD3+/CD8+/CD94-
    lymphocytes_CD3Pos_CD8Pos_CD94Neg_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD8', 'CD94', 1, 1, 0, 0, 1)[0]
    lymphocytes_CD3Pos_CD8Pos_CD94Neg_ratio = lymphocytes_CD3Pos_CD8Pos_CD94Neg_num / lymphocytes_CD3Pos_CD8Pos_num
    print('亚群Lymphocytes/CD3+/CD8+/CD94-的比率为', lymphocytes_CD3Pos_CD8Pos_CD94Neg_ratio)
    print('-'*100)
    ratio_list.append(lymphocytes_CD3Pos_CD8Pos_CD94Neg_ratio)

    # 56. Lymphocytes/CD3+/CD8+/CXCR5+
    lymphocytes_CD3Pos_CD8Pos_CXCR5Pos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD8', 'CXCR5', 1, 1, 0, 0, 0)[0]
    lymphocytes_CD3Pos_CD8Pos_CXCR5Pos_ratio = lymphocytes_CD3Pos_CD8Pos_CXCR5Pos_num / lymphocytes_CD3Pos_CD8Pos_num
    print('亚群Lymphocytes/CD3+/CD8+/CXCR5+的比率为', lymphocytes_CD3Pos_CD8Pos_CXCR5Pos_ratio)
    print('-'*100)
    ratio_list.append(lymphocytes_CD3Pos_CD8Pos_CXCR5Pos_ratio)

    # 59. Lymphocytes/CD3+/CD8+/CD278+
    lymphocytes_CD3Pos_CD8Pos_CD278Pos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD8', 'CD278', 1, 1, 0, 0, 0)[0]
    lymphocytes_CD3Pos_CD8Pos_CD278Pos_ratio = lymphocytes_CD3Pos_CD8Pos_CD278Pos_num / lymphocytes_CD3Pos_CD8Pos_num
    print('亚群Lymphocytes/CD3+/CD8+/CD278+的比率为', lymphocytes_CD3Pos_CD8Pos_CD278Pos_ratio)
    print('-'*100)
    ratio_list.append(lymphocytes_CD3Pos_CD8Pos_CD278Pos_ratio)

    # 60. Lymphocytes/CD3+/CD8+/PD1+
    lymphocytes_CD3Pos_CD8Pos_PD1Pos_num = subset_ratio.marker5_num('CD33', 'CD14', 'CD3', 'CD8', 'PD1', 1, 1, 0, 0, 0)[0]
    lymphocytes_CD3Pos_CD8Pos_PD1Pos_ratio = lymphocytes_CD3Pos_CD8Pos_PD1Pos_num / lymphocytes_CD3Pos_CD8Pos_num
    print('亚群Lymphocytes/CD3+/CD8+/PD1+的比率为', lymphocytes_CD3Pos_CD8Pos_PD1Pos_ratio)
    print('-'*100)
    ratio_list.append(lymphocytes_CD3Pos_CD8Pos_PD1Pos_ratio)

    # 61. Lymphocytes/CD3+/CD8+/Q1: 158Gd_CD197_CCR7- , 155Gd_CD45RA+
    CD8Pos_df = subset_ratio.marker4_num('CD33', 'CD14', 'CD3', 'CD8', 1, 1, 0, 0)[1]
    CD8Pos = SubsetsRatio(CD8Pos_df)
    CD197_CD45RA = CD8Pos.cross_classification('CD197', 'CD45RA')
    CD197Neg_CD45RAPos_num = CD197_CD45RA[3]
    CD197Neg_CD45RAPos_ratio = CD197Neg_CD45RAPos_num / lymphocytes_CD3Pos_CD8Pos_num
    # CD197Neg_CD45RAPos_ratio *= 2
    print('亚群Lymphocytes/CD3+/CD8+/Q1: 158Gd_CD197_CCR7- , 155Gd_CD45RA+的比率为', CD197Neg_CD45RAPos_ratio)
    print('-'*100)
    ratio_list.append(CD197Neg_CD45RAPos_ratio)

    # 62. Lymphocytes/CD3+/CD8+/Q2: 158Gd_CD197_CCR7+ , 155Gd_CD45RA+
    CD197Pos_CD45RAPos_num = CD197_CD45RA[0]
    CD197Pos_CD45RAPos_ratio = CD197Pos_CD45RAPos_num / lymphocytes_CD3Pos_CD8Pos_num
    # CD197Pos_CD45RAPos_ratio *= 1.3
    print('亚群Lymphocytes/CD3+/CD8+/Q2: 158Gd_CD197_CCR7+ , 155Gd_CD45RA+的比率为', CD197Pos_CD45RAPos_ratio)
    print('-'*100)
    ratio_list.append(CD197Pos_CD45RAPos_ratio)

    # 63. Lymphocytes/CD3+/CD8+/Q3: 158Gd_CD197_CCR7+ , 155Gd_CD45RA-
    CD197Pos_CD45RANeg_num = CD197_CD45RA[2]
    CD197Pos_CD45RANeg_ratio = CD197Pos_CD45RANeg_num / lymphocytes_CD3Pos_CD8Pos_num
    # CD197Pos_CD45RANeg_ratio /= 2
    print('亚群Lymphocytes/CD3+/CD8+/Q3: 158Gd_CD197_CCR7+ , 155Gd_CD45RA-的比率为', CD197Pos_CD45RANeg_ratio)
    print('-'*100)
    ratio_list.append(CD197Pos_CD45RANeg_ratio)

    # 64. Lymphocytes/CD3+/CD8+/Q4: 158Gd_CD197_CCR7- , 155Gd_CD45RA-
    CD197Neg_CD45RANeg_num = CD197_CD45RA[1]
    CD197Neg_CD45RANeg_ratio = CD197Neg_CD45RANeg_num / lymphocytes_CD3Pos_CD8Pos_num
    # CD197Neg_CD45RANeg_ratio /= 2
    print('亚群Lymphocytes/CD3+/CD8+/Q4: 158Gd_CD197_CCR7- , 155Gd_CD45RA-的比率为', CD197Neg_CD45RANeg_ratio)
    print('-'*100)
    ratio_list.append(CD197Neg_CD45RANeg_ratio)

    # 70. Lymphocytes/CD3-
    lymphocytes_CD3Neg_num = subset_ratio.marker3_num('CD33', 'CD14', 'CD3', 1, 1, 1)[0]
    lymphocytes_CD3Neg_ratio = lymphocytes_CD3Neg_num / lymphocytes_num
    print('亚群Lymphocytes/CD3-的比率为', lymphocytes_CD3Neg_ratio)
    print('-'*100)
    ratio_list.append(lymphocytes_CD3Neg_ratio)

    # 71. Lymphocytes/CD3-/B cells
    CD3Neg_df = subset_ratio.marker3_num('CD33', 'CD14', 'CD3', 1, 1, 1)[1]
    CD3Neg = SubsetsRatio(CD3Neg_df)
    # b_cells_num = CD3Neg.marker2_num('CD20', 'CD19', 0, 0)[0]
    b_cells_num = CD3Neg.marker1_num('CD19', 0)[0]
    b_cells_ratio = b_cells_num / lymphocytes_CD3Neg_num
    print('亚群Lymphocytes/CD3-/B cells的比率为', b_cells_ratio)
    print('-'*100)
    ratio_list.append(b_cells_ratio)

    # 81. Lymphocytes/CD3-/NK
    NK_num = CD3Neg.marker2_num('CD56', 'CD16', 0, 0)[0]
    NK_ratio = NK_num / lymphocytes_CD3Neg_num
    # NK_ratio *= 3
    print('亚群Lymphocytes/CD3-/NK的比率为', NK_ratio)
    print('-'*100)
    ratio_list.append(NK_ratio)

    # 82. Lymphocytes/CD3-/NK/CD57+
    NK_df = CD3Neg.marker2_num('CD56', 'CD16', 0, 0)[1]
    NK = SubsetsRatio(NK_df)
    CD57Pos_num = NK.marker1_num('CD57', 0)[0]
    CD57Pos_ratio = CD57Pos_num / NK_num
    print('亚群Lymphocytes/CD3-/NK/CD57+的比率为', CD57Pos_ratio)
    print('-'*100)
    ratio_list.append(CD57Pos_ratio)

    # 83. Lymphocytes/CD3-/NK/CD94+
    CD94Pos_num = NK.marker1_num('CD94', 0)[0]
    CD94Pos_ratio = CD94Pos_num / NK_num
    print('亚群Lymphocytes/CD3-/NK/CD94+的比率为', CD94Pos_ratio)
    print('-'*100)
    ratio_list.append(CD94Pos_ratio)

    # 84. Lymphocytes/CD3-/NK/CD94-
    CD94Neg_num = NK.marker1_num('CD94', 1)[0]
    CD94Neg_ratio = CD94Neg_num / NK_num
    print('亚群Lymphocytes/CD3-/NK/CD94-的比率为', CD94Neg_ratio)
    print('-'*100)
    ratio_list.append(CD94Neg_ratio)

    # 89. Lymphocytes/gd T-cells
    lymphocytes_df = subset_ratio.marker2_num('CD33', 'CD14', 1, 1)[1]
    Lymphocytes = SubsetsRatio(lymphocytes_df)
    gd_T_cells_num = Lymphocytes.marker2_num('gdTCR', 'CD3', 0, 0)[0]
    gd_T_cells_ratio = gd_T_cells_num / lymphocytes_num
    print('亚群Lymphocytes/gd T-cells的比率为', gd_T_cells_ratio)
    print('-'*100)
    ratio_list.append(gd_T_cells_ratio)

    # 90. Lymphocytes/NKT
    NK_T_num = Lymphocytes.marker2_num('CD3', 'CD56', 0, 0)[0]
    NK_T_ratio = NK_T_num / lymphocytes_num
    # NK_T_ratio *= 2
    print('亚群Lymphocytes/NKT的比率为', NK_T_ratio)
    print('-'*100)
    ratio_list.append(NK_T_ratio)

    # 91. Monocytes
    monocytes_df = difference_set(df, lymphocytes_df)
    monocytes_num = monocytes_df.shape[0]
    monocytes_ratio = monocytes_num / df.shape[0]
    print('亚群Monocytes的比率为', monocytes_ratio)
    print('-'*100)
    ratio_list.append(monocytes_ratio)

    # 92. Monocytes/CD14+CD16+
    Monocytes = SubsetsRatio(monocytes_df)
    CD14_CD16 = Monocytes.cross_classification('CD14', 'CD16')
    CD14Pos_CD16Pos_num = CD14_CD16[0]
    CD14Pos_CD16Pos_ratio = CD14Pos_CD16Pos_num / monocytes_num
    print('亚群Monocytes/CD14+CD16+的比率为', CD14Pos_CD16Pos_ratio)
    print('-'*100)
    ratio_list.append(CD14Pos_CD16Pos_ratio)

    # 93. Monocytes/CD14+CD16-
    CD14Pos_CD16Neg_num = CD14_CD16[2]
    CD14Pos_CD16Neg_ratio = CD14Pos_CD16Neg_num / monocytes_num
    print('亚群Monocytes/CD14+CD16-的比率为', CD14Pos_CD16Neg_ratio)
    print('-'*100)
    ratio_list.append(CD14Pos_CD16Neg_ratio)

    # 94. Monocytes/CD14-CD16+
    CD14Neg_CD16Pos_num = CD14_CD16[3]
    CD14Neg_CD16Pos_ratio = CD14Neg_CD16Pos_num / monocytes_num
    print('亚群Monocytes/CD14-CD16+的比率为', CD14Neg_CD16Pos_ratio)
    print('-'*100)
    ratio_list.append(CD14Neg_CD16Pos_ratio)

    # 95. Myeloid cells
    myeloid_num = subset_ratio.marker2_num('CD3', 'CD19', 1, 1)[0]
    myeloid_ratio = myeloid_num / len(df)
    print('亚群Myeloid cells的比率为', myeloid_ratio)
    print('-'*100)
    ratio_list.append(myeloid_ratio)

    # 96. Myeloid cells/CD56-CD14-
    myeloid_df = subset_ratio.marker2_num('CD3', 'CD19', 1, 1)[1]
    Myeloid = SubsetsRatio(myeloid_df)
    CD56Neg_CD14Neg_num = Myeloid.marker2_num('CD56', 'CD14', 1, 1)[0]
    CD56Neg_CD14Neg_ratio = CD56Neg_CD14Neg_num / myeloid_num
    print('亚群Myeloid cells/CD56-CD14-的比率为', CD56Neg_CD14Neg_ratio)
    print('-'*100)
    ratio_list.append(CD56Neg_CD14Neg_ratio)

    # 102. PD-L1+ cells
    PDL1_num = subset_ratio.marker1_num('CD274', 0)[0]
    PDL1_ratio = PDL1_num / df.shape[0]
    PDL1_ratio /= 100
    print('亚群PD-L1+ cells的比率为', PDL1_ratio)
    print('-'*100)
    ratio_list.append(PDL1_ratio)

    ratio_df = pd.DataFrame(ratio_list).T
    ratio_df.columns = ["Lymphocytes", "Lymphocytes/CD3+", "Lymphocytes/CD3+/CD4+", "Lymphocytes/CD3+/CD4+/CD57+", "Lymphocytes/CD3+/CD4+/CD94+", "Lymphocytes/CD3+/CD4+/CD94-", "Lymphocytes/CD3+/CD4+/ICOS+", "Lymphocytes/CD3+/CD4+/non-naive cells", "Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5+", "Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5-", "Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5-/CXCR3+CCR6+", "Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5-/Th1 CXCR3+CCR6-", "Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5-/Th2 CXCR3-CCR6-", "Lymphocytes/CD3+/CD4+/non-naive cells/CXCR5-/Th17 CXCR3-CCR6+", "Lymphocytes/CD3+/CD4+/PD1+", "Lymphocytes/CD3+/CD4+/Q1: 158Gd_CD197_CCR7- , 155Gd_CD45RA+", "Lymphocytes/CD3+/CD4+/Q2: 158Gd_CD197_CCR7+ , 155Gd_CD45RA+", "Lymphocytes/CD3+/CD4+/Q3: 158Gd_CD197_CCR7+ , 155Gd_CD45RA-", "Lymphocytes/CD3+/CD4+/Q4: 158Gd_CD197_CCR7- , 155Gd_CD45RA-", "Lymphocytes/CD3+/CD4+/Tfh", "Lymphocytes/CD3+/CD4+/Treg", "Lymphocytes/CD3+/CD8+", "Lymphocytes/CD3+/CD8+/CD57+", "Lymphocytes/CD3+/CD8+/CD94+", "Lymphocytes/CD3+/CD8+/CD94-", "Lymphocytes/CD3+/CD8+/CXCR5+", "Lymphocytes/CD3+/CD8+/ICOS+", "Lymphocytes/CD3+/CD8+/PD1+", "Lymphocytes/CD3+/CD8+/Q1: 158Gd_CD197_CCR7- , 155Gd_CD45RA+", "Lymphocytes/CD3+/CD8+/Q2: 158Gd_CD197_CCR7+ , 155Gd_CD45RA+", "Lymphocytes/CD3+/CD8+/Q3: 158Gd_CD197_CCR7+ , 155Gd_CD45RA-", "Lymphocytes/CD3+/CD8+/Q4: 158Gd_CD197_CCR7- , 155Gd_CD45RA-", "Lymphocytes/CD3-", "Lymphocytes/CD3-/B cells", "Lymphocytes/CD3-/NK cells", "Lymphocytes/CD3-/NK cells/CD57+", "Lymphocytes/CD3-/NK cells/CD94+", "Lymphocytes/CD3-/NK cells/CD94-", "Lymphocytes/gd T-cells", "Lymphocytes/NKT", "Monocytes", "Monocytes/CD14+CD16+", "Monocytes/CD14+CD16-", "Monocytes/CD14-CD16+", "Myeloid cells", "Myeloid cells/CD56-CD14-", "PD-L1+ cells"]
    return ratio_df
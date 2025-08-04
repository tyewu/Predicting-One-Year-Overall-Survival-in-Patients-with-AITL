# -*- coding: utf-8 -*-
# Author: wufatian

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import shap

from ._compute_auc_delong import *
from ..utils.graph import display_chn
from scipy.stats import mannwhitneyu

from sklearn.calibration import calibration_curve
import matplotlib.lines as mlines
# import matplotlib.transforms as mtransforms

class EvaModelFusion:

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. 

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def cal_predprob(self, clf, X_test):
        clf.fit(self.X_train, self.y_train)
        y_test_predprob = clf.predict_proba(X_test)[:, 1]
        return y_test_predprob

    def cal_res(self, clf, X_test, y_test, output=False, sm = False):
        """
        机器学习 模型训练训练集数据集后，在测试集输出评价结果，输出约登指数，并根据最优阈值，计算召回率，精确率等
        结果
        
        Args:
            clf: 机器学习分类模型
            X_train: 训练集数据
            y_train: 训练集数据标签
            X_test: 测试集数据
            y_test: 测试集数据标签
            output: 控制是否打印评价结果指标，默认不打印

        Returns:
            返回在测试集上计算的准确率，精确率，召回率，f1分数，auc结果，youden's index, y_test_predprob, fpr, tpr

        """
        
        clf.fit(self.X_train, self.y_train)
        y_test_preds = clf.predict(X_test)
        if sm == True:
            y_test_predprob = clf.decision_function(X_test)
        else:
            y_test_predprob = clf.predict_proba(X_test)[:,1]


        # 计算约登指数, youden's index
        fpr, tpr, thresholds = roc_curve(y_test, y_test_predprob, pos_label=1)
        yd = max(tpr - fpr)

        maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
        # 计算最佳阈值
        # thresholds[maxindex]

        y_test_preds = [1 if i >= thresholds[maxindex] else 0 for i in y_test_predprob]

        # 准确率
        acc = accuracy_score(y_test, y_test_preds)
        acc = round(acc, 4)
        # 精确率
        pre = precision_score(y_test, y_test_preds)
        pre = round(pre, 4)
        # 召回率
        rec = recall_score(y_test, y_test_preds)
        rec = round(rec, 4)
        # 特异度
        spe = (y_test[y_test == y_test_preds] == 0).sum()/(y_test == 0).sum()
        spe = round(spe, 4)
        # f1分数
        f1 = f1_score(y_test, y_test_preds)
        f1 = round(f1, 4)
        # auc
        roauc = roc_auc_score(y_test, y_test_predprob)
        roauc = round(roauc, 4)

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_test_preds)

        # delong auc 95%置信区间
        ci, auc_std = self.cal_delong(y_test, y_test_predprob)

        if output:
            print('Accuracy: {0:.4f}'.format(acc))
            print('Precision: {0:.4f}'.format(pre))
            print('Recall: {0:.4f}'.format(rec))
            print('f1_score: {0:.4f}'.format(f1))
            print('AUC: {0:.4f}'.format(auc))

        res = {}
        # res['model'] = clf
        res['acc'] = acc
        res['pre'] = pre
        res['rec'] = rec
        res['f1'] = f1
        res['roauc'] = roauc
        res['yd'] = yd
        res['y_test_predprob'] = y_test_predprob.tolist()
        res['fpr'] = fpr.tolist()
        res['tpr'] = tpr.tolist()
        res['ci'] = ci.tolist()
        res['auc_std'] = auc_std
        res['params'] = clf.get_params()
        res['spe'] = spe
        res['cm'] = cm
        # return acc, pre, rec, f1, roauc, yd, y_test_predprob, fpr, tpr, ci, auc_std
        return res

    def shap_plot(self, clf):
        """
        输出shap特征重要度结果，仅限于支持有限的机器学习模型

        Args:
            clf: 机器学习分类模型
            X_train: 训练集数据
            y_train: 训练集数据标签

        Returns:
            直接输出shap特征重要度结果
        """
        
        shap.initjs()
        display_chn()

        # clf.fit(self.X_train, self.y_train)
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(self.X_train)
        shap.summary_plot(shap_values, self.X_train, plot_type="bar")

    def multi_models_roc(self, names, sampling_methods, X_test, y_test, save=False, dpin=100):
        """
        将多个机器模型的roc图输出到一张图上
        
        Args:
            names: list, 多个模型的名称
            sampling_methods: list, 多个模型的实例化对象
            save: 选择是否将结果保存为PDF
        Returns:
            直接将图片显示出来
        """
        plt.figure(figsize=(20, 8), dpi=dpin)

        for (name, method) in zip(names, sampling_methods):
            res = self.cal_res(method, X_test, y_test, output=False)
            plt.plot(res['fpr'], res['tpr'], lw=2, label='{} (AUC={:.4f})'.format(name, auc(res['fpr'], res['tpr'])))
            plt.plot([0, 1], [0, 1], '--', lw=2)
            plt.axis('square')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.legend(fontsize=8)

        if save:
            plt.savefig('multi_models_roc.pdf')


    def model_roc(self, clf, X_test, y_test, dpin=100):
        plt.figure(figsize=(20, 8), dpi=dpin)
        res = self.cal_res(clf, X_test, y_test, output=False)
        plt.plot(res['fpr'], res['tpr'], lw=2, label='{} (AUC={:.4f})'.format('AUC', auc(res['fpr'], res['tpr'])))
        plt.plot([0, 1], [0, 1], '--', lw=2)
        plt.axis('square')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.legend(fontsize=8)



    def evalu_models(self, names, sampling_methods, X_test, y_test, save=False, svm=False):

        acc_dict = {}
        pre_dict = {}
        rec_dict = {}
        spe_dict = {}
        f1_dict = {}
        roauc_dict = {}
        yd_dict = {}
        auc_std_dict = {}
        auc_ci_dict = {}
        params_dict = {}
        pvalue_dict = {}
        cm_dict = {}

        a = svm
        for (name, method) in zip(names, sampling_methods):

            # acc, pre, rec, f1, roauc, yd, y_test_predprob, fpr, tpr = self.cal_res(method, X_test, y_test, output=False)
            res = self.cal_res(method, X_test, y_test, output=False, sm=a)
            acc_dict[name] = res["acc"]
            pre_dict[name] = res["pre"]
            rec_dict[name] = res["rec"]
            spe_dict[name] = res["spe"]
            f1_dict[name] = res["f1"]
            roauc_dict[name] = res["roauc"]
            yd_dict[name] = res["yd"]
            auc_ci_dict[name] = [round(res['ci'][0], 4), round(res['ci'][1], 4)]
            # auc_ci_dict[name] = list(res['ci'])
            auc_std_dict[name] = res['auc_std']
            params_dict[name] = res['params']
            pvalue_dict[name] = self.cal_auc_pvalue(y_test, res['y_test_predprob'])
            cm_dict[name] = res['cm']


        acc_ser = pd.Series(acc_dict)
        pre_ser = pd.Series(pre_dict)
        rec_ser = pd.Series(rec_dict)
        spe_ser = pd.Series(spe_dict)
        f1_ser = pd.Series(f1_dict)
        roauc_ser = pd.Series(roauc_dict)
        yd_ser = pd.Series(yd_dict)
        auc_ci_ser = pd.Series(auc_ci_dict)
        auc_std_ser = pd.Series(auc_std_dict)
        params_ser = pd.Series(params_dict)
        pvalue_ser = pd.Series(pvalue_dict)
        cm_ser = pd.Series(cm_dict)

        df = pd.DataFrame({
                           # 'params':params_ser,
                           'AUC':roauc_ser,
                           'AUC_95%CI':auc_ci_ser,
                           'AUC_SD':auc_std_ser,
                           'AUC_pvalue':pvalue_ser,
                           '准确率':acc_ser,
                           '精确率':pre_ser,
                           '召回率':rec_ser,
                           '特异度':spe_ser,
                           'f1分数':f1_ser,
                           '约登指数':yd_ser,
                           '混淆矩阵':cm_ser})

        if save:
            df.to_csv('evalu_model.csv', index=True)

        return df

    def features_plot(self, clf, save=False, dpin=100):
        """
        输出特征重要图
        1.importance_type=weight（默认值），特征重要性使用特征在所有树中作为划分属性的次数。
        2.importance_type=gain，特征重要性使用特征在作为划分属性时loss平均的降低量。
        3.importance_type=cover，特征重要性使用特征在作为划分属性时对样本的覆盖度。
        """
        feature = self.X_train.columns
        importance = clf.feature_importances_
        feature_importance = pd.Series(importance, index=feature).sort_values(ascending=False)
        # feature_importance = feature_importance[feature_importance.values!=0]
        sorted_names = feature_importance.index.values
        sorted_scores = feature_importance.values
        y_pos = np.arange(len(sorted_names))
        display_chn()
        plt.figure(figsize=(20, 20), dpi=dpin)
        fig, ax = plt.subplots(figsize=(15,15), dpi=dpin)
        ax.barh(y_pos, sorted_scores, height=0.7, align='center', color='green', tick_label=sorted_names)
        ax.set_yticks(y_pos)
        plt.yticks(fontsize=20)
        ax.set_xlabel('Feature Score')
        ax.set_ylabel('Features')
        ax.invert_yaxis()
        ax.set_title('Feature Importance')

        for score, pos in zip(sorted_scores, y_pos):
            ax.text(score + 0.005, pos, '%.5f' % score, ha='center', va='center', fontsize=16)

        plt.tight_layout()
        # plt.show()

        if save:
            plt.savefig('feature_importances.png')

        return list(sorted_names)

    def calibration_plot(self, clf, X_test, y_test, label_name, save=False, dpin=100):
        res = self.cal_res(clf, X_test, y_test, output=False)
        prob_true, prob_pred = calibration_curve(y_test, res['y_test_predprob'], n_bins=10)
        fig, ax = plt.subplots(figsize=(15,15), dpi=dpin)
        plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label=label_name)
        line = mlines.Line2D([0, 1], [0, 1], color='black')
        transform = ax.transAxes
        line.set_transform(transform)
        ax.add_line(line)
        fig.suptitle('Calibration plot for test data')
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('True probability in each bin')
        plt.legend()
        plt.show()
        if save:
            fig.savefig("calibration_plot.png")

    def cal_delong(self, y_test, y_test_predprob):
        alpha = .95
        auc, auc_cov = delong_roc_variance(y_test, y_test_predprob)
        auc_std = np.sqrt(auc_cov)
        lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
        ci = stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)
        ci[ci > 1] = 1
        return ci, auc_std

    def cal_auc_pvalue(self, y_test, y_test_predprob):
        obs = np.array(y_test)
        pred = np.array(y_test_predprob)
        res = mannwhitneyu(pred[obs==1], pred[obs==0], alternative='greater')
        return res.pvalue


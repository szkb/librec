/**
 * Copyright (C) 2016 LibRec
 * <p>
 * This file is part of LibRec.
 * LibRec is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * <p>
 * LibRec is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * <p>
 * You should have received a copy of the GNU General Public License
 * along with LibRec. If not, see <http://www.gnu.org/licenses/>.
 */
package net.librec.recommender.cf.rating;

import net.librec.common.LibrecException;
import net.librec.data.convertor.appender.AuxiliaryItemDataAppender;
import net.librec.math.structure.*;
import net.librec.math.structure.Vector;
import net.librec.recommender.MatrixFactorizationRecommender;
import net.librec.util.Lists;

import java.util.*;

/**
 * <ul>
 * <li><strong>PMF:</strong> Ruslan Salakhutdinov and Andriy Mnih, Probabilistic Matrix Factorization, NIPS 2008.</li>
 * <li><strong>RegSVD:</strong> Arkadiusz Paterek, <strong>Improving Regularized Singular Value Decomposition</strong>
 * Collaborative Filtering, Proceedings of KDD Cup and Workshop, 2007.</li>
 * </ul>
 *
 * @author guoguibin and zhanghaidong
 */
public class PMFItemFeatureRecommender extends MatrixFactorizationRecommender {

    private HashMap<Integer, ArrayList<Integer>> itemFeature;
    private double explicitWeight = 0.5;

    protected SequentialAccessSparseMatrix userInterestMatrix;
    protected SequentialAccessSparseMatrix socialMatrix;

    private int knn;
    private SymmMatrix similarityMatrix;
    private List<Map.Entry<Integer, Double>>[] userSimilarityList;
    private List<Integer> userList;
    private DenseVector userMeans;

    double[] meanRate;
    double[][] similarity;

    @Override
    protected void setup() throws LibrecException {
        super.setup();
        meanRate = new double[numUsers];
        similarity = new double[numUsers][numUsers];
        double[][] userInterest = new double[numUsers][19];

        // todo 类中的变量不会自动初始化吗

        knn = conf.getInt("rec.neighbors.knn.number");
        similarityMatrix = context.getSimilarity().getSimilarityMatrix();
        itemFeature = ((AuxiliaryItemDataAppender) getDataModel().getDataAppender()).getItemFeature();
//        socialMatrix = ((SocialDataAppender)  getDataModel().getDataAppender()).getUserAppender();
        userMeans = new VectorBasedDenseVector(numUsers);
        userList = new ArrayList<>(numUsers);
        for (int userIndex = 0; userIndex < numUsers; userIndex++) {
            userList.add(userIndex);
        }
        userList.parallelStream().forEach(userIndex -> {
            SequentialSparseVector userRatingVector = trainMatrix.row(userIndex);
            userMeans.set(userIndex, userRatingVector.getNumEntries() > 0 ? userRatingVector.mean() : globalMean);
        });

        createUserSimilarityList();


        for (int i = 0; i < numUsers; i++) {
            int[] ratedItemFeatureSum = new int[19];
            int featureSum = 0;
            SequentialSparseVector ratedItemVector = trainMatrix.row(i);
            for (Vector.VectorEntry simVectorEntry : ratedItemVector) {
                for (int j = 0; j < 19; j++) {
                    if (itemFeature.containsKey(Integer.valueOf(simVectorEntry.index()))
                            && j < itemFeature.get(Integer.valueOf(simVectorEntry.index())).size()) {
                        ratedItemFeatureSum[j] += itemFeature.get(Integer.valueOf(simVectorEntry.index())).get(j);
                    }
                }
                featureSum += featureSum(itemFeature.get(Integer.valueOf(simVectorEntry.index())));
            }
            if (featureSum > 0) {
                for (int k = 0; k < 19; k++) {
                    userInterest[i][k] = ratedItemFeatureSum[k] / featureSum * 1.0;
                }
            }
        }

        double[][] ratedItemSum = new double[numUsers][numItems];
        for (int i = 0; i < numUsers; i++) {
            for (int j = 0; j < numItems; j++) {
                for (int k = 0; k < 19; k++) {
                    if (userInterest[i][k] > 0 && itemFeature.get(j).get(k) > 0) {
                        ratedItemSum[i][j] += userInterest[i][k];
                    }
                }
            }
        }
    }

    public int featureSum(ArrayList<Integer> arrayList) {
        int sumFeature = 0;
        for (Integer feature : arrayList) {
            sumFeature += feature;
        }
        return sumFeature;
    }


    public void similarity(double[][] ratedItemSum) {
        for (int i = 0; i < numUsers; i++) {
            double temp = 0.0;
            int count = 0;
            for (int j = 0; j < numItems; j++) {
                if (ratedItemSum[i][j] > 0) {
                    temp += ratedItemSum[i][j];
                    count++;
                }
            }
            if (count > 0) {
                meanRate[i] = temp / count;
            }
        }


        for (int thisUser = 0; thisUser < numUsers; thisUser++) {
            double aboveSum = 0.0;
            double underSum = 0.0;
            double thisPow2 = 0.0;
            double thatPow2 = 0.0;
            for (int thatUser = thisUser + 1; thatUser < numUsers; thatUser++) {
                for (int j = 0; j < numItems; j++) {
                    double thisMinusMu = ratedItemSum[thisUser][j] - meanRate[thisUser];
                    double thatMinusMu = ratedItemSum[thatUser][j] - meanRate[thatUser];
                    aboveSum += thisMinusMu * thatMinusMu;
                    thisPow2 += thisMinusMu * thisMinusMu;
                    thatPow2 += thatMinusMu * thatMinusMu;
                }
                if (thisPow2 > 0 || thatPow2 > 0) {
                    similarity[thisUser][thatUser] = aboveSum / (Math.sqrt(thisPow2) * Math.sqrt(thatPow2));
                }
                ;
            }
        }

    }

    @Override
    protected void trainModel() throws LibrecException {
        for (int iter = 1; iter <= 70; iter++) {

            loss = 0.0d;
            for (MatrixEntry me : trainMatrix) {
                int userId = me.row(); // user
                int itemId = me.column(); // item
                double realRating = me.get();

                double predictRating = predict(userId, itemId);
                double error = realRating - predictRating;

                loss += error * error;

                // update factors
                for (int factorId = 0; factorId < numFactors; factorId++) {
                    double userFactor = userFactors.get(userId, factorId), itemFactor = itemFactors.get(itemId, factorId);

                    userFactors.plus(userId, factorId, learnRate * (error * itemFactor - regUser * userFactor));
                    itemFactors.plus(itemId, factorId, learnRate * (error * userFactor - regItem * itemFactor));

                    loss += regUser * userFactor * userFactor + regItem * itemFactor * itemFactor;
                }
            }

            loss *= 0.5;
            if (isConverged(iter) && earlyStop) {
                break;
            }
            updateLRate(iter);
        }
    }

    @Override
    protected double predict(int userIdx, int itemIdx) throws LibrecException {
        Map.Entry<Integer, Double> simUserEntry;
        double predictValue = 0.0D, simSum = 0.0D;

        double temp1 = explicitWeight * userFactors.row(userIdx).dot(itemFactors.row(itemIdx));
        List<Map.Entry<Integer, Double>> simList = userSimilarityList[userIdx];
        for (int i = 0; i < simList.size(); i++) {
            simUserEntry = simList.get(i);
//            predictValue += simUserEntry.getValue()
            if (userIdx < numUsers && simUserEntry.getKey() < numUsers) {
                predictValue += similarity[userIdx][simUserEntry.getKey().intValue()]
                        * userFactors.row(simUserEntry.getKey()).dot(itemFactors.row(itemIdx));
                // todo 相似度的综合
//            simSum += Math.abs(simUserEntry.getValue());
                simSum += Math.abs(similarity[userIdx][simUserEntry.getKey().intValue()]);
            }
        }

        double temp2 = 0;
        if (simSum > 0) {
            temp2 = (1 - explicitWeight) * predictValue / simSum;
        }
        return temp1 + temp2;
    }

    private void createUserSimilarityList() {
        userSimilarityList = new ArrayList[numUsers];
        SequentialAccessSparseMatrix simMatrix = similarityMatrix.toSparseMatrix();
        userList.parallelStream().forEach(userIndex -> {
            SequentialSparseVector similarityVector = simMatrix.row(userIndex);
            userSimilarityList[userIndex] = new ArrayList<>(similarityVector.size());
            for (Vector.VectorEntry simVectorEntry : similarityVector) {
                userSimilarityList[userIndex].add(new AbstractMap.SimpleImmutableEntry<>(simVectorEntry.index(), simVectorEntry.get()));
            }
            userSimilarityList[userIndex] = Lists.sortListTopK(userSimilarityList[userIndex], true, knn);
            Lists.sortListByKey(userSimilarityList[userIndex], false);
        });
    }
}

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
import net.librec.data.convertor.appender.AuxiliaryDataAppender;
import net.librec.data.model.ArffInstance;
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


public class PMFRecommender extends MatrixFactorizationRecommender {
    private static class ValueComparator implements Comparator<Map.Entry<Integer, Double>> {
        @Override
        public int compare(Map.Entry<Integer, Double> m, Map.Entry<Integer, Double> n) {
            if (n.getValue() - m.getValue() > 0) {
                return 1;
            } else if (n.getValue() - m.getValue() < 0) {
                return -1;
            } else {
                return 0;
            }
        }
    }

    // 用户相似度
    private int knn;
    private SymmMatrix similarityMatrix;
    private List<Map.Entry<Integer, Double>>[] userSimilarityList;
    private List<Integer> userList;
    private DenseVector userMeans;

    // 用户兴趣相似度数组
    double[][] similarity;
    double[][] similarityItem;
    // 用户自身对物品评分占比多少
    private double explicitWeight = 0.8;

    // 还原用户的原始ID
    private Map<Integer, String> userIdxToUserId;
    private Map<Integer, String> itemIdxToItemId;

    @Override
    protected void setup() throws LibrecException {
        super.setup();
        similarity = new double[numUsers][numUsers];
        similarityItem = new double[numItems][numItems];
        userIdxToUserId = context.getDataModel().getUserMappingData().inverse();
        itemIdxToItemId = context.getDataModel().getItemMappingData().inverse();
        try {
            readTag();
        } catch (Exception e) {
            e.printStackTrace();
        }

//        knn = conf.getInt("rec.neighbors.knn.number");
//        similarityMatrix = context.getSimilarity().getSimilarityMatrix();
//
//        userMeans = new VectorBasedDenseVector(numUsers);
//        userList = new ArrayList<>(numUsers);
//        for (int userIndex = 0; userIndex < numUsers; userIndex++) {
//            userList.add(userIndex);
//        }
//        userList.parallelStream().forEach(userIndex -> {
//            SequentialSparseVector userRatingVector = trainMatrix.row(userIndex);
//            userMeans.set(userIndex, userRatingVector.getNumEntries() > 0 ? userRatingVector.mean() : globalMean);
//        });

//        createUserSimilarityList();
//
        createUserTagSimilarityList();

//        createItemTagSimilarityList();


    }

    // 这里迭代次数有变化
    @Override
    protected void trainModel() throws LibrecException {
        for (int iter = 1; iter <= 150; iter++) {

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
                    Map.Entry<Integer, Double> simItemEntry;
                    double sumImpItemFactor = 0.0;
                    double sumSimilarity = 0.0;
                    double impItemAnswer = 0.0;
                    List<Map.Entry<Integer, Double>> simList = itemTagSimilarity.get(itemId);
                    for (int i = 0; i < simList.size(); i++){
                        simItemEntry = simList.get(i);
                        double impItemFactor = impItemFactors.get(simItemEntry.getKey(), factorId);
                        double impItemFactorValue = simItemEntry.getValue() * impItemFactors.get(simItemEntry.getKey(), factorId);
                        sumImpItemFactor += impItemFactorValue;
                        sumSimilarity += Math.abs(simItemEntry.getValue());
                        // todo 新增的参数
                        impItemFactors.plus(simItemEntry.getKey(), factorId, learnRate * ((1 - explicitWeight) * error * itemFactor - regUser * impItemFactor));
                        loss += regUser * impItemFactor * impItemFactor;
                    }
                    if (sumSimilarity > 0) {
                        impItemAnswer = sumImpItemFactor / sumSimilarity;
                    }

                    userFactors.plus(userId, factorId, learnRate * (error * explicitWeight * itemFactor - regUser * userFactor));
                    itemFactors.plus(itemId, factorId, learnRate * (error * (userFactor * explicitWeight + (1 - explicitWeight) * impItemAnswer) - regItem * itemFactor));


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
        Map.Entry<Integer, Double> simItemEntry;
        double predictValue = 0.0D, simSum = 0.0D;

        double temp1 = explicitWeight * userFactors.row(userIdx).dot(itemFactors.row(itemIdx));
//        List<Map.Entry<Integer, Double>> simList = userSimilarityList[userIdx];
        List<Map.Entry<Integer, Double>> simList = itemTagSimilarity.get(itemIdx);

        for (int i = 0; i < simList.size(); i++) {
            simItemEntry = simList.get(i);
            predictValue += simItemEntry.getValue()
                    // todo 这里还有些问题
//            predictValue += similarity[userIdx][simUserEntry.getKey()]
                    * impItemFactors.row(simItemEntry.getKey()).dot(itemFactors.row(itemIdx));
            // todo 相似度的综合
            simSum += Math.abs(simItemEntry.getValue());
//            simSum += Math.abs(similarity[userIdx][simUserEntry.getKey()]);

        }

        double temp2 = 0;
        if (simSum > 0) {
            temp2 = (1 - explicitWeight) * predictValue / simSum;
        }
        return temp1 + temp2;
    }


    /**
     * 创造用户标签相似度
     */
    private void createUserTagSimilarityList() {
        for (int thisUser = 0; thisUser < numUsers; thisUser++) {
            double aboveSum = 0.0;
            double thisPow2 = 0.0;
            double thatPow2 = 0.0;
            for (int thatUser = thisUser + 1; thatUser < numUsers; thatUser++) {
                String userA = userIdxToUserId.get(thisUser);
                String userB = userIdxToUserId.get(thatUser);
                Set<String> common = new HashSet<>();
                Set<String> userATag = new HashSet<>();
                if (userInformation.get(userA) != null) {
                    userATag =  userInformation.get(userA).keySet();
                }
                if (userInformation.get(userB) != null) {
                    for (String tag : userInformation.get(userB).keySet()) {
                        if (userATag.contains(tag)) {
                            common.add(tag);
                        }
                    }
                }

                for (String tag : common) {


                    double thisMinusMu = 0;
                    double thatMinusMu = 0;
                    if (userInformation.get(userA) != null && userInformation.get(userB) != null) {
                        if (userInformation.get(userA).get(tag) != null && userInformation.get(userB).get(tag) != null
                                && !Double.isNaN(meanTagNumber.get(userB)) && !Double.isNaN(meanTagNumber.get(userB))) {
                            thisMinusMu = userInformation.get(userA).get(tag) - meanTagNumber.get(userA);
                            thatMinusMu = userInformation.get(userB).get(tag) - meanTagNumber.get(userB);
                        }
                    }
                    aboveSum += thisMinusMu * thatMinusMu;
                    thisPow2 += thisMinusMu * thisMinusMu;
                    thatPow2 += thatMinusMu * thatMinusMu;
                }
                if (thisPow2 > 0 || thatPow2 > 0) {
                    // todo 要给它排序
                    similarity[thisUser][thatUser] = aboveSum / (Math.sqrt(thisPow2) * Math.sqrt(thatPow2));
                }
            }
        }

        rankUserTagSimilarityList(similarity);
    }

    private void createItemTagSimilarityList() {
        for (int thisItem = 0; thisItem < numItems; thisItem++) {
            double aboveSum = 0.0;
            double thisPow2 = 0.0;
            double thatPow2 = 0.0;
            for (int thatItem = thisItem + 1; thatItem < numItems; thatItem++) {
                String itemA = itemIdxToItemId.get(thisItem);
                String itemB = itemIdxToItemId.get(thatItem);
                Set<String> common = new HashSet<>();
                Set<String> itemATag = new HashSet<>();
                if (itemInformation.get(itemA) != null) {
                    itemATag =  itemInformation.get(itemA).keySet();
                }
                if (itemInformation.get(itemB) != null) {
                    for (String tag : itemInformation.get(itemB).keySet()) {
                        if (itemATag.contains(tag)) {
                            common.add(tag);
                        }
                    }
                }

                for (String tag : common) {


                    double thisMinusMu = 0;
                    double thatMinusMu = 0;
                    if (itemInformation.get(itemA) != null && itemInformation.get(itemB) != null) {
                        if (itemInformation.get(itemA).get(tag) != null && itemInformation.get(itemB).get(tag) != null
                                && !Double.isNaN(meanItemTagNumber.get(itemA)) && !Double.isNaN(meanItemTagNumber.get(itemB))) {
                            thisMinusMu = itemInformation.get(itemA).get(tag) - meanItemTagNumber.get(itemA);
                            thatMinusMu = itemInformation.get(itemB).get(tag) - meanItemTagNumber.get(itemB);
                        }
                    }
                    aboveSum += thisMinusMu * thatMinusMu;
                    thisPow2 += thisMinusMu * thisMinusMu;
                    thatPow2 += thatMinusMu * thatMinusMu;
                }
                if (thisPow2 > 0 || thatPow2 > 0) {
                    // todo 要给它排序
                    similarityItem[thisItem][thatItem] = aboveSum / (Math.sqrt(thisPow2) * Math.sqrt(thatPow2));
                }
            }
        }

        rankItemTagSimilarityList(similarityItem);
    }

    HashMap<Integer, List<Map.Entry<Integer, Double>>> userTagSimilarity = new HashMap<>();

    HashMap<Integer, List<Map.Entry<Integer, Double>>> itemTagSimilarity = new HashMap<>();


    private void rankUserTagSimilarityList(double[][] similarity) {
        for (int i = 0; i < numUsers; i++) {
            HashMap<Integer, Double> temp = new HashMap<>();
            for (int j = i + 1; j < numUsers; j++) {
                if (similarity[i][j] > 0) {
                    temp.put(j, similarity[i][j]);
                }
            }

            List<Map.Entry<Integer, Double>> list = new ArrayList<>();
            list.addAll(temp.entrySet());
            ValueComparator vc = new ValueComparator();
            Collections.sort(list, vc);

            List<Map.Entry<Integer, Double>> listTemmp = new ArrayList<>();
            for (int k = 0; k < list.size(); k++) {
                if (k < 20) {
                    listTemmp.add(list.get(k));
                } else {
                    break;
                }
            }
            userTagSimilarity.put(i, listTemmp);

        }
    }

    private void rankItemTagSimilarityList(double[][] similarityItem) {
        for (int i = 0; i < numItems; i++) {
            HashMap<Integer, Double> temp = new HashMap<>();
            for (int j = i + 1; j < numItems; j++) {
                if (similarityItem[i][j] > 0) {
                    temp.put(j, similarityItem[i][j]);
                }
            }

            List<Map.Entry<Integer, Double>> list = new ArrayList<>();
            list.addAll(temp.entrySet());
            ValueComparator vc = new ValueComparator();
            Collections.sort(list, vc);

            List<Map.Entry<Integer, Double>> listTemmp = new ArrayList<>();
            for (int k = 0; k < list.size(); k++) {
                if (k < 20) {
                    listTemmp.add(list.get(k));
                } else {
                    break;
                }
            }
            itemTagSimilarity.put(i, listTemmp);

        }
    }
    //
//    /**
//     * 求用户相似度
//     */
//
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

    // 保存全部标签
    HashSet<String> set = new HashSet<>();
    HashSet<String> setItem = new HashSet<>();

    /**
     * 读取数据
     */
    public void readTag() {
        ArrayList<ArffInstance> auxiliaryData = ((AuxiliaryDataAppender) getDataModel().getDataAppender()).getAuxiliaryData();

        System.out.println("**************");
        System.out.println(auxiliaryData.size());
        HashMap<String, ArrayList<String>> userTagInformation = new HashMap<>();
        HashMap<String, ArrayList<String>> itemTagInformation = new HashMap<>();

        for (ArffInstance instance : auxiliaryData) {
            String userId = (String) instance.getValueByIndex(0);
            String movieId = (String) instance.getValueByIndex(1);
            String tag = (String) instance.getValueByIndex(2);
            String timestamp = (String) instance.getValueByIndex(3);

            // 保存全部标签
            set.add(tag);
            setItem.add(movieId);
            if (!userTagInformation.containsKey(userId)) {
                ArrayList<String> arrayList = new ArrayList<>();
                arrayList.add(tag);
                userTagInformation.put(userId, arrayList);
            } else {
                ArrayList<String> arrayList = new ArrayList<>();
                arrayList.add(tag);
                arrayList.addAll(userTagInformation.get(userId));
                userTagInformation.put(userId, arrayList);
            }

            if (!itemTagInformation.containsKey(movieId)) {
                ArrayList<String> arrayList = new ArrayList<>();
                arrayList.add(tag);
                itemTagInformation.put(movieId, arrayList);
            } else {
                ArrayList<String> arrayList = new ArrayList<>();
                arrayList.add(tag);
                arrayList.addAll(itemTagInformation.get(movieId));
                itemTagInformation.put(movieId, arrayList);
            }
        }
        // 解析数据
        parseUserTagInformation(userTagInformation);
        parseItemTagInformation(itemTagInformation);
    }

    // 保存评价的平均标签数和所有的用户信息
    HashMap<String, Double> meanTagNumber = new HashMap<>();
    HashMap<String, HashMap<String, Integer>> userInformation = new HashMap<>();

    HashMap<String, Double> meanItemTagNumber = new HashMap<>();
    HashMap<String, HashMap<String, Integer>> itemInformation = new HashMap<>();


    public void parseUserTagInformation(HashMap<String, ArrayList<String>> userTagInformation) {
        for (String user : userTagInformation.keySet()) {
            ArrayList<String> tagInformation = userTagInformation.get(user);
            HashMap<String, Integer> tagNumber = new HashMap<>();

            for (String tag : tagInformation) {
                if (!tagNumber.containsKey(tag)) {
                    tagNumber.put(tag, 1);
                } else {
                    int count = tagNumber.get(tag) + 1;
                    tagNumber.put(tag, count);
                }
            }

            // 每一个用户的标签信息存入进去
            userInformation.put(user, tagNumber);

            // 求每一个用户使用标签的平均次数
            int sumTagNumber = 0;
            for (Integer value : tagNumber.values()) {
                sumTagNumber += value;
            }
            if (tagNumber.size() > 0) {
                meanTagNumber.put(user, sumTagNumber * 1.0 / tagNumber.size());
            }

        }

    }


    public void parseItemTagInformation(HashMap<String, ArrayList<String>> itemTagInformation) {
        for (String item : itemTagInformation.keySet()) {
            ArrayList<String> tagInformation = itemTagInformation.get(item);
            HashMap<String, Integer> tagNumber = new HashMap<>();

            for (String tag : tagInformation) {
                if (!tagNumber.containsKey(tag)) {
                    tagNumber.put(tag, 1);
                } else {
                    int count = tagNumber.get(tag) + 1;
                    tagNumber.put(tag, count);
                }
            }

            // 每一个用户的标签信息存入进去
            itemInformation.put(item, tagNumber);

            // 求每一个用户使用标签的平均次数
            int sumTagNumber = 0;
            for (Integer value : tagNumber.values()) {
                sumTagNumber += value;
            }
            if (tagNumber.size() > 0) {
                meanItemTagNumber.put(item, sumTagNumber * 1.0 / tagNumber.size());
            }

        }

    }

}

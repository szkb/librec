package net.librec.recommender.cf.rating;

import net.librec.common.LibrecException;
import net.librec.data.convertor.appender.AuxiliaryDataAppender;
import net.librec.data.model.ArffInstance;
import net.librec.math.structure.*;
import net.librec.math.structure.Vector;
import net.librec.recommender.MatrixFactorizationRecommender;
import net.librec.util.Lists;

import java.util.*;

public class PMFUserRecommender extends MatrixFactorizationRecommender {
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

    HashMap<Integer, List<Map.Entry<Integer, Double>>> userTagSimilarity = new HashMap<>();
    // 保存评价的平均标签数和所有的用户信息
    HashMap<String, Double> meanTagNumber = new HashMap<>();
    HashMap<String, HashMap<String, Integer>> userInformation = new HashMap<>();


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

        createUserTagSimilarityList();



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
//        List<Map.Entry<Integer, Double>> simList = userSimilarityList[userIdx];
        List<Map.Entry<Integer, Double>> simList = userTagSimilarity.get(userIdx);

        for (int i = 0; i < simList.size(); i++) {
            simUserEntry = simList.get(i);
//            predictValue += simUserEntry.getValue()
            if (userIdx < numUsers && simUserEntry.getKey() < numUsers) {
                predictValue += simUserEntry.getValue()
                        * userFactors.row(simUserEntry.getKey()).dot(itemFactors.row(itemIdx));
                // todo 相似度的综合
                simSum += Math.abs(simUserEntry.getValue());
            }
        }

        double temp2 = 0;
        if (simSum > 0) {
            temp2 = (1 - explicitWeight) * predictValue / simSum;
            return temp1 + temp2;
        }
        return temp1 / explicitWeight;
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
        HashMap<String, ArrayList<String>> userTagInformation = new HashMap<>();

        for (ArffInstance instance : auxiliaryData) {
            String userId = (String) instance.getValueByIndex(0);
            String movieId = (String) instance.getValueByIndex(1);
            String tag = (String) instance.getValueByIndex(2);

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

        }
        // 解析数据
        parseUserTagInformation(userTagInformation);
    }




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
}

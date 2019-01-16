package net.librec.recommender.cf.rating;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import net.librec.common.LibrecException;
import net.librec.data.convertor.appender.AuxiliaryUserTagAppender;
import net.librec.data.convertor.appender.TagDataAppender;
import net.librec.math.structure.*;
import net.librec.math.structure.Vector;
import net.librec.recommender.MatrixFactorizationRecommender;

import java.util.*;

import static net.librec.recommender.cf.rating.PMFBigItemRecommender.CAPACITY;

/**
 * @author szkb
 * @date 2019/01/11 21:45
 */
public class PMFMatrixRecommender extends MatrixFactorizationRecommender {
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

    public static final int NUMBER = 100000;

    private HashMap<Integer, ArrayList<Integer>> dislikeSetUser = new HashMap<>();
    private HashMap<Integer, ArrayList<Integer>> neutralSetUser = new HashMap<>();
    private HashMap<Integer, ArrayList<Integer>> likeSetUser = new HashMap<>();

    private HashMap<Integer, Double> mapRate = new HashMap<>();
    protected SequentialAccessSparseMatrix tagMatrix;

    protected SequentialAccessSparseMatrix itemTagGradeMatrix;
    Table<Integer, Integer, Double> dataTable = HashBasedTable.create();

    protected SequentialAccessSparseMatrix posTagGradeMatrix;
    Table<Integer, Integer, Double> dataTablePos = HashBasedTable.create();

    protected SequentialAccessSparseMatrix midTagGradeMatrix;
    Table<Integer, Integer, Double> dataTableMid = HashBasedTable.create();

    protected SequentialAccessSparseMatrix negTagGradeMatrix;
    Table<Integer, Integer, Double> dataTableNeg = HashBasedTable.create();


    // 用户兴趣相似度数组
    private double[][] similarity;
    private double[][] similarityItem;
    // 用户自身对物品评分占比多少
    private double explicitWeight = 0.8;

    private double posWeight = 0.3;
    private double negWeight = 0.3;

    // 还原用户的原始ID
    private Map<Integer, String> userIdxToUserId;
    private Map<Integer, String> itemIdxToItemId;

    private double[] mean;

    @Override
    protected void setup() throws LibrecException {
        super.setup();
        tagMatrix = ((TagDataAppender) getDataModel().getDataAppender()).getUserAppender();
        similarity = new double[numUsers][numUsers];
        similarityItem = new double[numItems][numItems];
        userIdxToUserId = context.getDataModel().getUserMappingData().inverse();
        itemIdxToItemId = context.getDataModel().getItemMappingData().inverse();

        mean = new double[numUsers];
        square();
        classify(trainMatrix);
        parseTagInformation(tagMatrix);

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
                    Map.Entry<Integer, Double> simUserEntry;
                    double sumImpUserFactor = 0.0;
                    double sumSimilarity = 0.0;
                    double impUserAnswer = 0.0;
                    List<Map.Entry<Integer, Double>> simList = userTagSimilarity.get(userId);
                    for (int i = 0; i < simList.size(); i++) {
                        simUserEntry = simList.get(i);
                        double impUserFactor = impUserFactors.get(simUserEntry.getKey(), factorId);
                        double impUserFactorValue = simUserEntry.getValue() * impUserFactors.get(simUserEntry.getKey(), factorId);
                        sumImpUserFactor += impUserFactorValue;
                        sumSimilarity += Math.abs(simUserEntry.getValue());
                        // todo 新增的参数
                        impUserFactors.plus(simUserEntry.getKey(), factorId, learnRate * ((1 - explicitWeight) * error * itemFactor - regUser * impUserFactor));
                        loss += regUser * impUserFactor * impUserFactor;
                    }
                    if (sumSimilarity > 0) {
                        impUserAnswer = sumImpUserFactor / sumSimilarity;
                    }

                    userFactors.plus(userId, factorId, learnRate * (error * explicitWeight * itemFactor - regUser * userFactor));
                    itemFactors.plus(itemId, factorId, learnRate * (error * (userFactor * explicitWeight + (1 - explicitWeight) * impUserAnswer) - regItem * itemFactor));


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
            predictValue += simUserEntry.getValue()
                    // todo 这里还有些问题
//            predictValue += similarity[userIdx][simUserEntry.getKey()]
                    * impUserFactors.row(simUserEntry.getKey()).dot(itemFactors.row(itemIdx));
            // todo 相似度的综合
            simSum += Math.abs(simUserEntry.getValue());
//            simSum += Math.abs(similarity[userIdx][simUserEntry.getKey()]);

        }

        double temp2 = 0;
        if (simSum > 0) {
            temp2 = (1 - explicitWeight) * predictValue / simSum;
            return temp1 + temp2;
        }
        return temp1 / explicitWeight;
    }


    //<editor-fold desc="用户相似度">

    /**
     * 创造用户标签相似度
     */
    private void createUserTagSimilarityList() {
        for (int thisUser = 0; thisUser < numUsers; thisUser++) {

            for (int thatUser = thisUser + 1; thatUser < numUsers; thatUser++) {

                //<editor-fold desc="正项目相似度">
                Set<Integer> commonPos = new HashSet<>();
                if (posTagGradeMatrix.row(thisUser).size() > 0 && posTagGradeMatrix.row(thatUser).size() > 0) {
                    int[] thisNums = posTagGradeMatrix.row(thisUser).getIndices();
                    int[] thatNums = posTagGradeMatrix.row(thatUser).getIndices();
                    for (int i = 0; i < thisNums.length; i++) {
                        for (int j = 0; j < thatNums.length; j++) {
                            if (thisNums[i] == thatNums[j]) {
                                commonPos.add(thisNums[i]);
                            }
                        }

                    }
                }
                if (commonPos.size() == 0) {
                    continue;
                }
                double above = 0.0;
                double underThis = 0.0;
                double underThat = 0.0;

                for (Integer tag : commonPos) {
                    if (posTagGradeMatrix.get(thisUser, tag) > 0 && posTagGradeMatrix.get(thatUser, tag) > 0) {
                        double thisTagGrade = posTagGradeMatrix.get(thisUser, tag);
                        double thatTagGrade = posTagGradeMatrix.get(thatUser, tag);
                        above += thisTagGrade * thatTagGrade;

                        underThis += thisTagGrade * thisTagGrade;
                        underThat += thatTagGrade * thatTagGrade;

                    }

                }

                double ans = above / (Math.sqrt(underThis) * Math.sqrt(underThat));
                //</editor-fold>
                similarity[thisUser][thatUser] = ans;

            }
        }

        for (int thisUser = 0; thisUser < numUsers; thisUser++) {

            for (int thatUser = thisUser + 1; thatUser < numUsers; thatUser++) {
                //<editor-fold desc="负项目相似度">
                Set<Integer> commonNeg = new HashSet<>();
                if (negTagGradeMatrix.row(thisUser).size() > 0 && negTagGradeMatrix.row(thatUser).size() > 0) {
                    int[] thisNums = negTagGradeMatrix.row(thisUser).getIndices();
                    int[] thatNums = negTagGradeMatrix.row(thatUser).getIndices();
                    for (int i = 0; i < thisNums.length; i++) {
                        for (int j = 0; j < thatNums.length; j++) {
                            if (thisNums[i] == thatNums[j]) {
                                commonNeg.add(thisNums[i]);
                            }
                        }

                    }
                }
                if (commonNeg.size() == 0) {
                    continue;
                }
                double aboveNeg = 0.0;
                double underThisNeg = 0.0;
                double underThatNeg = 0.0;

                for (Integer tag : commonNeg) {
                    if (negTagGradeMatrix.get(thisUser, tag) > 0 && negTagGradeMatrix.get(thatUser, tag) > 0) {
                        double thisTagGrade = negTagGradeMatrix.get(thisUser, tag);
                        double thatTagGrade = negTagGradeMatrix.get(thatUser, tag);
                        aboveNeg += thisTagGrade * thatTagGrade;

                        underThisNeg += thisTagGrade * thisTagGrade;
                        underThatNeg += thatTagGrade * thatTagGrade;

                    }

                }

                double ansNeg = aboveNeg / (Math.sqrt(underThisNeg) * Math.sqrt(underThatNeg));
                //</editor-fold>

                similarity[thisUser][thatUser] = similarity[thisUser][thatUser] + ansNeg;

            }
        }

        for (int thisUser = 0; thisUser < numUsers; thisUser++) {

            for (int thatUser = thisUser + 1; thatUser < numUsers; thatUser++) {
                //<editor-fold desc="负项目相似度">
                Set<Integer> commonMid = new HashSet<>();
                if (midTagGradeMatrix.row(thisUser).size() > 0 && midTagGradeMatrix.row(thatUser).size() > 0) {
                    int[] thisNums = midTagGradeMatrix.row(thisUser).getIndices();
                    int[] thatNums = midTagGradeMatrix.row(thatUser).getIndices();
                    for (int i = 0; i < thisNums.length; i++) {
                        for (int j = 0; j < thatNums.length; j++) {
                            if (thisNums[i] == thatNums[j]) {
                                commonMid.add(thisNums[i]);
                            }
                        }

                    }
                }
                if (commonMid.size() == 0) {
                    continue;
                }
                double aboveMid = 0.0;
                double underThisMid = 0.0;
                double underThatMid = 0.0;

                for (Integer tag : commonMid) {
                    if (midTagGradeMatrix.get(thisUser, tag) > 0 && midTagGradeMatrix.get(thatUser, tag) > 0) {
                        double thisTagGrade = midTagGradeMatrix.get(thisUser, tag);
                        double thatTagGrade = midTagGradeMatrix.get(thatUser, tag);
                        aboveMid += thisTagGrade * thatTagGrade;

                        underThisMid += thisTagGrade * thisTagGrade;
                        underThatMid += thatTagGrade * thatTagGrade;

                    }

                }

                double ansMid = aboveMid / (Math.sqrt(underThisMid) * Math.sqrt(underThatMid));
                //</editor-fold>

                similarity[thisUser][thatUser] = (similarity[thisUser][thatUser] + ansMid) / 3;

            }
        }


        rankUserTagSimilarityList(similarity);
    }

//    private void createItemTagSimilarityList() {
//        for (int thisItem = 0; thisItem < numItems; thisItem++) {
//            double aboveSum = 0.0;
//            double thisPow2 = 0.0;
//            double thatPow2 = 0.0;
//            for (int thatItem = thisItem + 1; thatItem < numItems; thatItem++) {
//                String itemA = itemIdxToItemId.get(thisItem);
//                String itemB = itemIdxToItemId.get(thatItem);
//                Set<String> common = new HashSet<>();
//                Set<String> itemATag = new HashSet<>();
//                if (itemInformation.get(itemA) != null) {
//                    itemATag = itemInformation.get(itemA).keySet();
//                }
//                if (itemInformation.get(itemB) != null) {
//                    for (String tag : itemInformation.get(itemB).keySet()) {
//                        if (itemATag.contains(tag)) {
//                            common.add(tag);
//                        }
//                    }
//                }
//
//                for (String tag : common) {
//
//
//                    double thisMinusMu = 0;
//                    double thatMinusMu = 0;
//                    if (itemInformation.get(itemA) != null && itemInformation.get(itemB) != null) {
//                        if (itemInformation.get(itemA).get(tag) != null && itemInformation.get(itemB).get(tag) != null
//                                && !Double.isNaN(meanItemTagNumber.get(itemA)) && !Double.isNaN(meanItemTagNumber.get(itemB))) {
//                            thisMinusMu = itemInformation.get(itemA).get(tag) - meanItemTagNumber.get(itemA);
//                            thatMinusMu = itemInformation.get(itemB).get(tag) - meanItemTagNumber.get(itemB);
//                        }
//                    }
//                    aboveSum += thisMinusMu * thatMinusMu;
//                    thisPow2 += thisMinusMu * thisMinusMu;
//                    thatPow2 += thatMinusMu * thatMinusMu;
//                }
//                if (thisPow2 > 0 || thatPow2 > 0) {
//                    // todo 要给它排序
//                    similarityItem[thisItem][thatItem] = aboveSum / (Math.sqrt(thisPow2) * Math.sqrt(thatPow2));
//                }
//            }
//        }
//
////        rankItemTagSimilarityList(similarityItem);
//    }

    HashMap<Integer, List<Map.Entry<Integer, Double>>> userTagSimilarity = new HashMap<>();

//    HashMap<Integer, List<Map.Entry<Integer, Double>>> itemTagSimilarity = new HashMap<>();


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

//    private void rankItemTagSimilarityList(double[][] similarityItem) {
//        for (int i = 0; i < numItems; i++) {
//            HashMap<Integer, Double> temp = new HashMap<>();
//            for (int j = i + 1; j < numItems; j++) {
//                if (similarityItem[i][j] > 0) {
//                    temp.put(j, similarityItem[i][j]);
//                }
//            }
//
//            List<Map.Entry<Integer, Double>> list = new ArrayList<>();
//            list.addAll(temp.entrySet());
//            ValueComparator vc = new ValueComparator();
//            Collections.sort(list, vc);
//
//            List<Map.Entry<Integer, Double>> listTemmp = new ArrayList<>();
//            for (int k = 0; k < list.size(); k++) {
//                if (k < 20) {
//                    listTemmp.add(list.get(k));
//                } else {
//                    break;
//                }
//            }
//            itemTagSimilarity.put(i, listTemmp);
//
//        }
//    }
    //</editor-fold>

    //<editor-fold desc="解析标签信息求平均标签数">
//    // 保存全部标签
//    HashSet<String> set = new HashSet<>();
//    HashSet<String> setItem = new HashSet<>();
//
//
//    // 保存评价的平均标签数和所有的用户信息
//    HashMap<String, Double> meanTagNumber = new HashMap<>();
//    HashMap<String, HashMap<String, Integer>> userInformation = new HashMap<>();
//
//    HashMap<String, Double> meanItemTagNumber = new HashMap<>();
//
//    public void parseUserTagInformation(HashMap<String, ArrayList<String>> userTagInformation) {
//        for (String user : userTagInformation.keySet()) {
//            ArrayList<String> tagInformation = userTagInformation.get(user);
//            HashMap<String, Integer> tagNumber = new HashMap<>();
//
//            for (String tag : tagInformation) {
//                if (!tagNumber.containsKey(tag)) {
//                    tagNumber.put(tag, 1);
//                } else {
//                    int count = tagNumber.get(tag) + 1;
//                    tagNumber.put(tag, count);
//                }
//            }
//
//            // 每一个用户的标签信息存入进去
//            userInformation.put(user, tagNumber);
//
//            // 求每一个用户使用标签的平均次数
//            int sumTagNumber = 0;
//            for (Integer value : tagNumber.values()) {
//                sumTagNumber += value;
//            }
//            if (tagNumber.size() > 0) {
//                meanTagNumber.put(user, sumTagNumber * 1.0 / tagNumber.size());
//            }
//
//        }
//
//    }
//
//    public void parseItemTagInformation(HashMap<String, ArrayList<String>> itemTagInformation) {
//        for (String item : itemTagInformation.keySet()) {
//            ArrayList<String> tagInformation = itemTagInformation.get(item);
//            HashMap<String, Integer> tagNumber = new HashMap<>();
//
//            for (String tag : tagInformation) {
//                if (!tagNumber.containsKey(tag)) {
//                    tagNumber.put(tag, 1);
//                } else {
//                    int count = tagNumber.get(tag) + 1;
//                    tagNumber.put(tag, count);
//                }
//            }
//
//            // 每一个用户的标签信息存入进去
//            itemInformation.put(item, tagNumber);
//
//            // 求每一个用户使用标签的平均次数
//            int sumTagNumber = 0;
//            for (Integer value : tagNumber.values()) {
//                sumTagNumber += value;
//            }
//            if (tagNumber.size() > 0) {
//                meanItemTagNumber.put(item, sumTagNumber * 1.0 / tagNumber.size());
//            }
//
//        }
//
//    }
//    //</editor-fold>


//    HashMap<String, HashMap<String, Integer>> itemInformation = new HashMap<>();
//
//    private HashMap<Integer, HashMap<String, Double>> preferences = new HashMap<>();
//    private HashMap<Integer, HashMap<String, Double>> preferencesNeg = new HashMap<>();
//
//    private HashMap<Integer, ArrayList<String>> posTag = new HashMap<>();
//    private HashMap<Integer, ArrayList<String>> negTag = new HashMap<>();
//    private HashMap<Integer, ArrayList<String>> midTag = new HashMap<>();

    public void parseTagInformation(SequentialAccessSparseMatrix tagMatrix) {
        for (int userId = 0; userId < numUsers; userId++) {
            for (Vector.VectorEntry itemId : tagMatrix.row(userId)) {
                if (trainMatrix.get(userId, itemId.index()) > 0) {
                    double temp = 0.0;
                    if (mapRate.containsKey(userId)) {
                        temp = trainMatrix.get(userId, itemId.index()) / mapRate.get(userId);
                        int tag = (int) itemId.get();
                        dataTable.put(itemId.index(), tag, temp);

                    }
                }
            }
        }

        itemTagGradeMatrix = new SequentialAccessSparseMatrix(NUMBER, NUMBER, dataTable);


        for (int i = 0; i < numUsers; i++) {
            // 保存在正的项目中每个标签出现的项目总数
            ArrayList<Integer> itemPosList = likeSetUser.get(i);
            HashMap<Integer, Integer> tagPosNumbers = new HashMap<>();

            ArrayList<Integer> itemMidList = neutralSetUser.get(i);
            HashMap<Integer, Integer> tagMidNumbers = new HashMap<>();
            // 保存在负的项目中每个标签出现的项目总数
            ArrayList<Integer> itemNegList = dislikeSetUser.get(i);
            HashMap<Integer, Integer> tagNegNumbers = new HashMap<>();

            // 保存每个标签分别在正负项目用户的偏好程度
            HashMap<Integer, Double> tagPosGrade = new HashMap<>();
            HashMap<Integer, Double> tagMidGrade = new HashMap<>();
            HashMap<Integer, Double> tagNegGrade = new HashMap<>();

            // 用户是否有评分信息
//            HashMap<Integer, HashMap<String, Double>> itemGrade = new HashMap<>();
//            itemGrade = tagWeight.get(i);
//            if (itemGrade == null) {
//                continue;
//            }


            //<editor-fold desc="遍历正项目集合">
            if (itemPosList != null) {
                for (Integer item : itemPosList) {
                    // 项目有评分，但是不一定有标签，所以先要判断一下
                    if (itemTagGradeMatrix.row(item).size() > 0) {
                        for (Vector.VectorEntry tagId : itemTagGradeMatrix.row(item)) {
                            if (!tagPosGrade.containsKey(tagId.index())) {
                                // todo
                                tagPosGrade.put(tagId.index(), tagId.get());

                            } else {
                                tagPosGrade.put(tagId.index(), tagPosGrade.get(tagId.index()) + tagId.get());
                            }

                            if (tagPosNumbers.containsKey(tagId.index())) {
                                tagPosNumbers.put(tagId.index(), tagPosNumbers.get(tagId.index()) + 1);
                            } else {
                                tagPosNumbers.put(tagId.index(), 1);
                            }
                        }
                    }


                }
            }
            //</editor-fold>

            //<editor-fold desc="遍历负项目集合">
            if (itemNegList != null) {
                for (Integer item : itemNegList) {
                    // 项目有评分，但是不一定有标签，所以先要判断一下
                    if (itemTagGradeMatrix.row(item).size() > 0) {
                        for (Vector.VectorEntry tagId : itemTagGradeMatrix.row(item)) {
                            if (!tagNegGrade.containsKey(tagId.index())) {
                                // todo
                                tagNegGrade.put(tagId.index(), tagId.get());

                            } else {
                                tagNegGrade.put(tagId.index(), tagNegGrade.get(tagId.index()) + tagId.get());
                            }

                            if (tagNegNumbers.containsKey(tagId.index())) {
                                tagNegNumbers.put(tagId.index(), tagNegNumbers.get(tagId.index()) + 1);
                            } else {
                                tagNegNumbers.put(tagId.index(), 1);
                            }
                        }
                    }


                }
            }
            //</editor-fold>

            //<editor-fold desc="遍历中性项目集合">
            if (itemMidList != null) {
                for (Integer item : itemMidList) {

                    // 项目有评分，但是不一定有标签
                    if (itemTagGradeMatrix.row(item).size() > 0) {
                        for (Vector.VectorEntry tagId : itemTagGradeMatrix.row(item)) {
                            if (!tagMidGrade.containsKey(tagId.index())) {
                                // todo
                                tagMidGrade.put(tagId.index(), tagId.get());

                            } else {
                                tagMidGrade.put(tagId.index(), tagMidGrade.get(tagId.index()) + tagId.get());
                            }

                            if (tagMidNumbers.containsKey(tagId.index())) {
                                tagMidNumbers.put(tagId.index(), tagMidNumbers.get(tagId.index()) + 1);
                            } else {
                                tagMidNumbers.put(tagId.index(), 1);
                            }
                        }

                    }


                }
            }


            //</editor-fold>
            for (Integer tagPos : tagPosGrade.keySet()) {
                dataTablePos.put(i, tagPos, tagPosGrade.get(tagPos) / tagPosNumbers.get(tagPos));
            }
            for (Integer tagMid : tagMidGrade.keySet()) {
                dataTableMid.put(i, tagMid, tagMidGrade.get(tagMid) / tagMidNumbers.get(tagMid));
            }

            for (Integer tagNeg : tagNegGrade.keySet()) {
                dataTableNeg.put(i, tagNeg, tagNegGrade.get(tagNeg) / tagNegNumbers.get(tagNeg));
            }

        }

        posTagGradeMatrix = new SequentialAccessSparseMatrix(NUMBER, NUMBER, dataTablePos);
        midTagGradeMatrix = new SequentialAccessSparseMatrix(NUMBER, NUMBER, dataTableMid);
        negTagGradeMatrix = new SequentialAccessSparseMatrix(NUMBER, NUMBER, dataTableNeg);
        dataTablePos = null;
        dataTableMid = null;
        dataTableNeg = null;
    }

    public void classify(SequentialAccessSparseMatrix trainMatrix) {
        for (MatrixEntry me : trainMatrix) {
            int userId = me.row(); // user
            int itemId = me.column(); // item
            double realRating = me.get();

            if (mean[userId] > 0) {
                if (realRating > 0 && realRating <= 2) {
                    if (!dislikeSetUser.containsKey(userId)) {
                        ArrayList<Integer> arrayList = new ArrayList<>();
                        arrayList.add(itemId);
                        dislikeSetUser.put(userId, arrayList);
                    } else {
                        ArrayList<Integer> arrayList = new ArrayList<>();
                        arrayList.add(itemId);

                        ArrayList<Integer> listTemp = dislikeSetUser.get(userId);
                        arrayList.addAll(listTemp);

                        dislikeSetUser.put(userId, arrayList);

                    }
                } else if (realRating > 2 && realRating <= 3.5) {
                    if (!neutralSetUser.containsKey(userId)) {
                        ArrayList<Integer> arrayList = new ArrayList<>();
                        arrayList.add(itemId);
                        neutralSetUser.put(userId, arrayList);
                    } else {
                        ArrayList<Integer> arrayList = new ArrayList<>();
                        arrayList.add(itemId);

                        ArrayList<Integer> listTemp = neutralSetUser.get(userId);
                        arrayList.addAll(listTemp);

                        neutralSetUser.put(userId, arrayList);

                    }
                } else {
                    if (!likeSetUser.containsKey(userId)) {
                        ArrayList<Integer> arrayList = new ArrayList<>();
                        arrayList.add(itemId);
                        likeSetUser.put(userId, arrayList);
                    } else {
                        ArrayList<Integer> arrayList = new ArrayList<>();
                        arrayList.add(itemId);

                        ArrayList<Integer> listTemp = likeSetUser.get(userId);
                        arrayList.addAll(listTemp);

                        likeSetUser.put(userId, arrayList);

                    }
                }
            }


        }

    }


    public void square() {
        for (int i = 0; i < numUsers; i++) {
            double sum1 = 0.0;
            double sum2 = 0.0;
            int[] item = trainMatrix.row(i).getIndices();
            for (int j = 0; j < item.length; j++) {
                if (trainMatrix.get(i, j) > 0) {
                    sum1 += Math.pow(trainMatrix.get(i, j), 2);
                    sum2 += trainMatrix.get(i, j);
                }
            }
            if (item.length > 0) {
                mean[i] = sum2 / item.length;
            }
            if (sum1 > 0) {
                mapRate.put(i, Math.sqrt(sum1));
            }
        }
    }
}

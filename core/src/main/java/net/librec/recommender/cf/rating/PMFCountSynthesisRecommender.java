package net.librec.recommender.cf.rating;

import net.librec.common.LibrecException;
import net.librec.data.convertor.appender.AuxiliaryUserTagAppender;
import net.librec.math.structure.MatrixEntry;
import net.librec.math.structure.SequentialAccessSparseMatrix;
import net.librec.recommender.MatrixFactorizationRecommender;

import java.util.*;

import static net.librec.recommender.cf.rating.PMFBigItemRecommender.CAPACITY;

/**
 * @author szkb
 * @date 2019/01/19 19:19
 */
public class PMFCountSynthesisRecommender extends MatrixFactorizationRecommender {


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

    private HashMap<Integer, HashMap<Integer, ArrayList<String>>> tagInformation = new HashMap<>(CAPACITY);
    private HashMap<Integer, ArrayList<Integer>> likeSetUser = new HashMap<>();

    private HashMap<Integer, HashMap<Integer, ArrayList<String>>> tagItemInformation = new HashMap<>(CAPACITY);
    private HashMap<Integer, ArrayList<Integer>> likeItemSetUser = new HashMap<>();

    private HashMap<Integer, HashMap<Integer, HashMap<String, Double>>> tagWeight = new HashMap<>();
    private HashMap<Integer, Double> mapRate = new HashMap<>();

    private HashMap<Integer, HashMap<Integer, HashMap<String, Double>>> tagItemWeight = new HashMap<>();
    private HashMap<Integer, Double> mapRateItem = new HashMap<>();


    // 用户兴趣相似度数组
    private double[][] similarity;

    //    private double[][] maxSimilarity;
    private double[][] similarityItem;
    // 用户自身对物品评分占比多少
    private double explicitWeight = 0.8;

    private double userWeight = 0.2 * (1 - explicitWeight);

    private double itemWeight = 0.8 * (1 - explicitWeight);

    // 还原用户的原始ID
    private Map<Integer, String> userIdxToUserId;
    private Map<Integer, String> itemIdxToItemId;

    private double[] mean;
    private double[] meanItem;

    @Override
    protected void setup() throws LibrecException {
        super.setup();
        tagInformation = ((AuxiliaryUserTagAppender) getDataModel().getDataAppender()).getTagInformation();
        tagItemInformation = ((AuxiliaryUserTagAppender) getDataModel().getDataAppender()).getTagItemInformation();


        similarity = new double[numUsers][numUsers];
        similarityItem = new double[numItems][numItems];
        userIdxToUserId = context.getDataModel().getUserMappingData().inverse();
        itemIdxToItemId = context.getDataModel().getItemMappingData().inverse();

        mean = new double[numUsers];
        meanItem = new double[numItems];
        square();
        classify(trainMatrix);
        parseTagInformation(tagInformation);
        createUserTagSimilarityList();

        squareItem();
        classifyItem(trainMatrix);
        parseItemTagInformation(tagItemInformation);
        createItemTagSimilarityList();

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
                        double impUserFactor = userFactors.get(simUserEntry.getKey(), factorId);
                        double impUserFactorValue = simUserEntry.getValue() * impUserFactor;
                        sumImpUserFactor += impUserFactorValue;
                        sumSimilarity += Math.abs(simUserEntry.getValue());
//                        // todo 新增的参数
//                        impItemFactors.plus(simItemEntry.getKey(), factorId, learnRate * ((1 - explicitWeight) * error * itemFactor - regUser * impItemFactor));
//                        loss += regUser * impItemFactor * impItemFactor;
                    }
                    if (sumSimilarity > 0) {
                        impUserAnswer = sumImpUserFactor / sumSimilarity;
                    }

                    Map.Entry<Integer, Double> simItemEntry;
                    double sumImpItemFactor = 0.0;
                    double sumItemSimilarity = 0.0;
                    double impItemAnswer = 0.0;
                    List<Map.Entry<Integer, Double>> simItemList = itemTagSimilarity.get(itemId);
                    for (int j = 0; j < simItemList.size(); j++) {
                        simItemEntry = simItemList.get(j);
                        double impItemFactor = itemFactors.get(simItemEntry.getKey(), factorId);
                        double impItemFactorValue = simItemEntry.getValue() * impItemFactor;
                        sumImpItemFactor += impItemFactorValue;
                        sumItemSimilarity += Math.abs(simItemEntry.getValue());
//                        // todo 新增的参数
//                        impItemFactors.plus(simItemEntry.getKey(), factorId, learnRate * ((1 - explicitWeight) * error * itemFactor - regUser * impItemFactor));
//                        loss += regUser * impItemFactor * impItemFactor;
                    }
                    if (sumItemSimilarity > 0) {
                        impItemAnswer = sumImpItemFactor / sumItemSimilarity;
                    }


                    userFactors.plus(userId, factorId, learnRate * (error * itemFactor - regUser * userFactor));
                    itemFactors.plus(itemId, factorId, learnRate * (error * (userFactor * explicitWeight + userWeight * impUserAnswer + itemWeight * impItemAnswer) - regItem * itemFactor));

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
        Map.Entry<Integer, Double> simItemEntry;
        double predictValueUser = 0.0D, simUserSum = 0.0D;
        double predictValueItem = 0.0D, simItemSum = 0.0D;

        double temp1 = explicitWeight * userFactors.row(userIdx).dot(itemFactors.row(itemIdx));
//        List<Map.Entry<Integer, Double>> simList = userSimilarityList[userIdx];
        List<Map.Entry<Integer, Double>> simList = userTagSimilarity.get(userIdx);

        for (int i = 0; i < simList.size(); i++) {
            simUserEntry = simList.get(i);
            if (userIdx < numUsers && simUserEntry.getKey() < numUsers) {
                predictValueUser += simUserEntry.getValue()
                        * userFactors.row(simUserEntry.getKey()).dot(itemFactors.row(itemIdx));
                // todo 相似度的综合
                simUserSum += Math.abs(simUserEntry.getValue());
            }
        }

        List<Map.Entry<Integer, Double>> simItemList = itemTagSimilarity.get(itemIdx);

        for (int j = 0; j < simItemList.size(); j++) {
            simItemEntry = simItemList.get(j);
            if (itemIdx < numItems && simItemEntry.getKey() < numItems) {
                predictValueItem += simItemEntry.getValue()
                        * itemFactors.row(simItemEntry.getKey()).dot(itemFactors.row(itemIdx));
                // todo 相似度的综合
                simItemSum += Math.abs(simItemEntry.getValue());
            }
        }

        double temp2 = 0;
        if (simUserSum > 0) {
            temp2 = userWeight * predictValueUser / simUserSum;
        }

        double temp3 = 0;
        if (simItemSum > 0) {
            temp3 = itemWeight * predictValueItem / simItemSum;
        }
        return temp1 + temp2 + temp3;
    }


    //<editor-fold desc="用户相似度">

    /**
     * 创造用户标签相似度
     */
    private void createUserTagSimilarityList() {
        for (int thisUser = 0; thisUser < numUsers; thisUser++) {

            for (int thatUser = thisUser + 1; thatUser < numUsers; thatUser++) {

                //<editor-fold desc="正项目相似度">
                Set<String> commonPos = new HashSet<>();
                if (posTag.containsKey(thisUser) && posTag.containsKey(thatUser)) {
                    for (String thisTag : posTag.get(thisUser)) {
                        if (posTag.get(thatUser).contains(thisTag)) {
                            commonPos.add(thisTag);
                        }
                    }
                }
                if (commonPos.size() == 0) {
                    continue;
                }
                double above = 0.0;
                double underThis = 0.0;
                double underThat = 0.0;

                for (String tag : commonPos) {
                    if (preferences.containsKey(thisUser) && preferences.containsKey(thatUser)) {
                        double thisTagGrade = preferences.get(thisUser).get(tag);
                        double thatTagGrade = preferences.get(thatUser).get(tag);
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
        rankUserTagSimilarityList(similarity);
    }

    HashMap<Integer, List<Map.Entry<Integer, Double>>> userTagSimilarity = new HashMap<>();

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


    //</editor-fold>


    public void parseTagInformation(HashMap<Integer, HashMap<Integer, ArrayList<String>>> tagInformation) {
        for (Integer userId : tagInformation.keySet()) {
            HashMap<Integer, HashMap<String, Double>> itemTagGrade = new HashMap<>();
            for (Integer itemId : tagInformation.get(userId).keySet()) {
                if (trainMatrix.get(userId, itemId) > 0) {
                    HashMap<String, Double> tagGrade = new HashMap<>();
                    for (String tag : tagInformation.get(userId).get(itemId)) {
                        if (!tagAmount.containsKey(tag)) {
                            tagAmount.put(tag, 1);
                        } else {
                            int count = tagAmount.get(tag) + 1;
                            tagAmount.put(tag, count);
                        }
                        double temp = 0.0;
                        if (mapRate.containsKey(userId)) {
                            temp = trainMatrix.get(userId, itemId) / mapRate.get(userId);
                        }
                        tagGrade.put(tag, temp);
                    }
                    itemTagGrade.put(itemId, tagGrade);
                }
            }
            tagWeight.put(userId, itemTagGrade);
        }

        int sum = 0;
        for (String tag : tagAmount.keySet()) {
            sum += tagAmount.get(tag);
        }
        for (Map.Entry<String, Integer> entry : tagAmount.entrySet()) {
            double TFWeight = entry.getValue() * 1.0 / sum;
            tagTF.put(entry.getKey(), TFWeight);
        }

        for (int i = 0; i < numUsers; i++) {

            // 保存在正的项目中每个标签出现的项目总数
            ArrayList<Integer> itemPosList = likeSetUser.get(i);
            HashMap<String, Integer> tagPosNumbers = new HashMap<>();

            // 保存每个标签分别在正负项目用户的偏好程度
            HashMap<String, Double> tagPosGrade = new HashMap<>();

            // 用户是否有评分信息
            HashMap<Integer, HashMap<String, Double>> itemGrade = new HashMap<>();
            itemGrade = tagWeight.get(i);
            if (itemGrade == null) {
                continue;
            }


            //<editor-fold desc="遍历正项目集合">
            if (itemPosList != null) {
                for (Integer item : itemPosList) {

                    HashMap<String, Double> tagPoints = new HashMap<>();

                    // 项目有评分，但是不一定有标签，所以先要判断一下
                    if (itemGrade.containsKey(item)) {
                        tagPoints = itemGrade.get(item);
                    }


                    for (String tag : tagPoints.keySet()) {

                        if (!tagPosGrade.containsKey(tag)) {
                            // todo
                            tagPosGrade.put(tag, tagPoints.get(tag));

                        } else {
                            tagPosGrade.put(tag, tagPosGrade.get(tag) + tagPoints.get(tag));
                        }

                        if (tagPosNumbers.containsKey(tag)) {
                            tagPosNumbers.put(tag, tagPosNumbers.get(tag) + 1);
                        } else {
                            tagPosNumbers.put(tag, 1);
                        }
                    }


                }
            }
            //</editor-fold>


            Set<String> allTag = new HashSet<>();
            HashMap<String, Double> allTagGrade = new HashMap<>();
            ArrayList<String> arrayListPos = new ArrayList<>();


            for (String tag : tagPosGrade.keySet()) {
                arrayListPos.add(tag);
                double ans = tagPosGrade.get(tag) / tagPosNumbers.get(tag) * tagTF.get(tag);
                tagPosGrade.put(tag, ans);
                allTag.add(tag);
            }


            for (String tag : allTag) {

                allTagGrade.put(tag, tagPosGrade.get(tag));

            }
            // todo 还要做负的
            if (arrayListPos.size() > 0) {
                posTag.put(i, arrayListPos);
            }


            if (allTagGrade.size() > 0) {
                preferences.put(i, allTagGrade);
            }

            allTag = null;
            allTagGrade = null;
            arrayListPos = null;
        }


    }


    public void classify(SequentialAccessSparseMatrix trainMatrix) {
        for (MatrixEntry me : trainMatrix) {
            int userId = me.row(); // user
            int itemId = me.column(); // item
            double realRating = me.get();

            if (mean[userId] > 0) {

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

    private void createItemTagSimilarityList() {
        for (int thisItem = 0; thisItem < numItems; thisItem++) {

            for (int thatItem = thisItem + 1; thatItem < numItems; thatItem++) {

                //<editor-fold desc="正项目相似度">
                Set<String> commonPos = new HashSet<>();
                if (posItemTag.containsKey(thisItem) && posItemTag.containsKey(thatItem)) {
                    for (String thisTag : posItemTag.get(thisItem)) {
                        if (posItemTag.get(thatItem).contains(thisTag)) {
                            commonPos.add(thisTag);
                        }
                    }
                }
                if (commonPos.size() == 0) {
                    continue;
                }
                double above = 0.0;
                double underThis = 0.0;
                double underThat = 0.0;

                for (String tag : commonPos) {
                    if (preferencesItem.containsKey(thisItem) && preferencesItem.containsKey(thatItem)) {
                        double thisTagGrade = preferencesItem.get(thisItem).get(tag);
                        double thatTagGrade = preferencesItem.get(thatItem).get(tag);
                        above += thisTagGrade * thatTagGrade;

                        underThis += thisTagGrade * thisTagGrade;
                        underThat += thatTagGrade * thatTagGrade;

                    }

                }

                double ans = above / (Math.sqrt(underThis) * Math.sqrt(underThat));
                //</editor-fold>
//                similarity[thisItem][thatItem] = Math.max(ans, similarity[thisItem][thatItem]);
                similarityItem[thisItem][thatItem] = ans;

            }
        }


        rankItemTagSimilarityList(similarityItem);
    }

    HashMap<Integer, List<Map.Entry<Integer, Double>>> itemTagSimilarity = new HashMap<>();

    private void rankItemTagSimilarityList(double[][] similarity) {
        for (int i = 0; i < numItems; i++) {
            HashMap<Integer, Double> temp = new HashMap<>();
            for (int j = i + 1; j < numItems; j++) {
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
            itemTagSimilarity.put(i, listTemmp);

        }
    }


    //</editor-fold>


    private HashMap<Integer, HashMap<String, Double>> preferences = new HashMap<>();
    private HashMap<Integer, ArrayList<String>> posTag = new HashMap<>();

    private HashMap<Integer, HashMap<String, Double>> preferencesItem = new HashMap<>();
    private HashMap<Integer, ArrayList<String>> posItemTag = new HashMap<>();

    HashMap<String, Integer> tagAmount = new HashMap<>();
    HashMap<String, Double> tagTF = new HashMap<>();

    HashMap<String, Integer> tagItemAmount = new HashMap<>();
    HashMap<String, Double> tagItemTF = new HashMap<>();

    public void parseItemTagInformation(HashMap<Integer, HashMap<Integer, ArrayList<String>>> tagItemInformation) {
        for (Integer itemId : tagItemInformation.keySet()) {
            HashMap<Integer, HashMap<String, Double>> userTagGrade = new HashMap<>();
            for (Integer userId : tagItemInformation.get(itemId).keySet()) {
                if (trainMatrix.get(userId, itemId) > 0) {
                    HashMap<String, Double> tagGrade = new HashMap<>();
                    for (String tag : tagItemInformation.get(itemId).get(userId)) {
                        if (!tagItemAmount.containsKey(tag)) {
                            tagItemAmount.put(tag, 1);
                        } else {
                            int count = tagItemAmount.get(tag) + 1;
                            tagItemAmount.put(tag, count);
                        }
                        double temp = 0.0;
                        if (mapRateItem.containsKey(itemId)) {
                            temp = trainMatrix.get(userId, itemId) / mapRateItem.get(itemId);
                        }
                        tagGrade.put(tag, temp);
                    }
                    userTagGrade.put(userId, tagGrade);
                }
            }
            tagItemWeight.put(itemId, userTagGrade);
        }

        int sum = 0;
        for (String tag : tagItemAmount.keySet()) {
            sum += tagItemAmount.get(tag);
        }
        for (Map.Entry<String, Integer> entry : tagItemAmount.entrySet()) {
            double TFWeight = entry.getValue() * 1.0 / sum;
            tagItemTF.put(entry.getKey(), TFWeight);
        }

        for (int i = 0; i < numItems; i++) {

            // 保存在正的项目中每个标签出现的项目总数
            ArrayList<Integer> userPosList = likeItemSetUser.get(i);
            HashMap<String, Integer> tagPosNumbers = new HashMap<>();


            // 保存每个标签分别在正负项目用户的偏好程度
            HashMap<String, Double> tagPosGrade = new HashMap<>();


            // 用户是否有评分信息
            HashMap<Integer, HashMap<String, Double>> userGrade = new HashMap<>();
            userGrade = tagItemWeight.get(i);
            if (userGrade == null) {
                continue;
            }


            //<editor-fold desc="遍历正项目集合">
            if (userPosList != null) {
                for (Integer user : userPosList) {

                    HashMap<String, Double> tagPoints = new HashMap<>();

                    // 项目有评分，但是不一定有标签，所以先要判断一下
                    if (userGrade.containsKey(user)) {
                        tagPoints = userGrade.get(user);
                    }


                    for (String tag : tagPoints.keySet()) {

                        if (!tagPosGrade.containsKey(tag)) {
                            // todo
                            tagPosGrade.put(tag, tagPoints.get(tag));

                        } else {
                            tagPosGrade.put(tag, tagPosGrade.get(tag) + tagPoints.get(tag));
                        }

                        if (tagPosNumbers.containsKey(tag)) {
                            tagPosNumbers.put(tag, tagPosNumbers.get(tag) + 1);
                        } else {
                            tagPosNumbers.put(tag, 1);
                        }
                    }


                }
            }
            //</editor-fold>


            Set<String> allTag = new HashSet<>();
            HashMap<String, Double> allTagGrade = new HashMap<>();
            ArrayList<String> arrayListPos = new ArrayList<>();


            for (String tag : tagPosGrade.keySet()) {
                arrayListPos.add(tag);
                double ans = tagPosGrade.get(tag) / tagPosNumbers.get(tag) * tagItemTF.get(tag);
                tagPosGrade.put(tag, ans);
                allTag.add(tag);
            }


            for (String tag : allTag) {

                allTagGrade.put(tag, tagPosGrade.get(tag));

            }
            // todo 还要做负的
            if (arrayListPos.size() > 0) {
                posItemTag.put(i, arrayListPos);
            }


            if (allTagGrade.size() > 0) {
                preferencesItem.put(i, allTagGrade);
            }

            allTag = null;
            allTagGrade = null;
            arrayListPos = null;
        }


    }


    public void classifyItem(SequentialAccessSparseMatrix trainMatrix) {
        for (MatrixEntry me : trainMatrix) {
            int userId = me.row();
            int itemId = me.column();
            double realRating = me.get();

            if (meanItem[itemId] > 0) {

                if (!likeItemSetUser.containsKey(itemId)) {
                    ArrayList<Integer> arrayList = new ArrayList<>();
                    arrayList.add(userId);
                    likeItemSetUser.put(itemId, arrayList);
                } else {
                    ArrayList<Integer> arrayList = new ArrayList<>();
                    arrayList.add(userId);

                    ArrayList<Integer> listTemp = likeItemSetUser.get(itemId);
                    arrayList.addAll(listTemp);

                    likeItemSetUser.put(itemId, arrayList);

                }
            }


        }

    }


    public void squareItem() {
        for (int i = 0; i < numItems; i++) {
            double sum1 = 0.0;
            double sum2 = 0.0;

            // column与row
            int[] user = trainMatrix.column(i).getIndices();
            for (int j = 0; j < user.length; j++) {
                if (trainMatrix.get(j, i) > 0) {
                    sum1 += Math.pow(trainMatrix.get(j, i), 2);
                    sum2 += trainMatrix.get(j, i);
                }
            }
            if (user.length > 0) {
                meanItem[i] = sum2 / user.length;
            }
            if (sum1 > 0) {
                mapRateItem.put(i, Math.sqrt(sum1));
            }
        }
    }
}

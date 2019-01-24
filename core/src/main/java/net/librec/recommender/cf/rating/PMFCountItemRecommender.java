package net.librec.recommender.cf.rating;

import net.librec.common.LibrecException;
import net.librec.data.convertor.appender.AuxiliaryItemTagAppender;
import net.librec.math.structure.MatrixEntry;
import net.librec.math.structure.SequentialAccessSparseMatrix;
import net.librec.recommender.MatrixFactorizationRecommender;

import java.util.*;

import static net.librec.recommender.cf.rating.PMFBigItemRecommender.CAPACITY;

/**
 * @author szkb
 * @date 2019/01/18 15:45
 */
public class PMFCountItemRecommender extends MatrixFactorizationRecommender {

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
    private HashMap<Integer, HashMap<Integer, HashMap<String, Double>>> tagWeight = new HashMap<>();
    private HashMap<Integer, Double> mapRate = new HashMap<>();

    //    private double[][] maxSimilarity;
    // 项目相似度
    private double[][] similarityItem;
    // 用户自身对物品评分占比多少
    private double explicitWeight = 0.8;

    private double posWeight = 0.3;
    private double negWeight = 0.3;
    private static int count = 0;


    private double[] mean;

    @Override
    protected void setup() throws LibrecException {
        super.setup();
        tagInformation = ((AuxiliaryItemTagAppender) getDataModel().getDataAppender()).getTagInformation();
//        maxSimilarity = new double[numUsers][numUsers];
        similarityItem = new double[numItems][numItems];
        mean = new double[numItems];

        square();
        classify(trainMatrix);
        parseTagInformation(tagInformation);
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
                    Map.Entry<Integer, Double> simItemEntry;
                    double sumImpItemFactor = 0.0;
                    double sumSimilarity = 0.0;
                    double impItemAnswer = 0.0;
                    List<Map.Entry<Integer, Double>> simList = itemTagSimilarity.get(itemId);
                    for (int i = 0; i < simList.size(); i++) {
                        simItemEntry = simList.get(i);
                        double impItemFactor = itemFactors.get(simItemEntry.getKey(), factorId);
                        double impItemFactorValue = simItemEntry.getValue() * impItemFactor;
                        sumImpItemFactor += impItemFactorValue;
                        sumSimilarity += Math.abs(simItemEntry.getValue());
//                        // todo 新增的参数
//                        impItemFactors.plus(simItemEntry.getKey(), factorId, learnRate * ((1 - explicitWeight) * error * itemFactor - regUser * impItemFactor));
//                        loss += regUser * impItemFactor * impItemFactor;
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
                    * itemFactors.row(simItemEntry.getKey()).dot(itemFactors.row(itemIdx));
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


    //<editor-fold desc="用户相似度">

    /**
     * 创造用户标签相似度
     */
    private void createItemTagSimilarityList() {
        for (int thisItem = 0; thisItem < numItems; thisItem++) {

            for (int thatItem = thisItem + 1; thatItem < numItems; thatItem++) {

                //<editor-fold desc="正项目相似度">
                Set<String> commonPos = new HashSet<>();
                if (posTag.containsKey(thisItem) && posTag.containsKey(thatItem)) {
                    for (String thisTag : posTag.get(thisItem)) {
                        if (posTag.get(thatItem).contains(thisTag)) {
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
                    if (preferences.containsKey(thisItem) && preferences.containsKey(thatItem)) {
                        double thisTagGrade = preferences.get(thisItem).get(tag);
                        double thatTagGrade = preferences.get(thatItem).get(tag);
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
    HashMap<String, Integer> tagAmount = new HashMap<>();
    HashMap<String, Double> tagTF = new HashMap<>();

    public void parseTagInformation(HashMap<Integer, HashMap<Integer, ArrayList<String>>> tagInformation) {
        for (Integer itemId : tagInformation.keySet()) {
            HashMap<Integer, HashMap<String, Double>> userTagGrade = new HashMap<>();
            for (Integer userId : tagInformation.get(itemId).keySet()) {
                if (trainMatrix.get(userId, itemId) > 0) {
                    HashMap<String, Double> tagGrade = new HashMap<>();
                    for (String tag : tagInformation.get(itemId).get(userId)) {

                        if (!tagAmount.containsKey(tag)) {
                            tagAmount.put(tag, 1);
                        } else {
                            int count = tagAmount.get(tag) + 1;
                            tagAmount.put(tag, count);
                        }

                        double temp = 0.0;
                        if (mapRate.containsKey(itemId)) {
                            temp = trainMatrix.get(userId, itemId) / mapRate.get(itemId);
                        }
                        tagGrade.put(tag, temp);
                    }
                    userTagGrade.put(userId, tagGrade);
                }
            }
            tagWeight.put(itemId, userTagGrade);
        }

        int sum = 0;
        for (String tag : tagAmount.keySet()) {
            sum += tagAmount.get(tag);
        }
        for (Map.Entry<String, Integer> entry : tagAmount.entrySet()) {
            double TFWeight = entry.getValue() * 1.0 / sum;
            tagTF.put(entry.getKey(), TFWeight);
        }

        for (int i = 0; i < numItems; i++) {

            // 保存在正的项目中每个标签出现的项目总数
            ArrayList<Integer> userPosList = likeSetUser.get(i);
            HashMap<String, Integer> tagPosNumbers = new HashMap<>();


            // 保存每个标签分别在正负项目用户的偏好程度
            HashMap<String, Double> tagPosGrade = new HashMap<>();


            // 用户是否有评分信息
            HashMap<Integer, HashMap<String, Double>> userGrade = new HashMap<>();
            userGrade = tagWeight.get(i);
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
            int userId = me.row();
            int itemId = me.column();
            double realRating = me.get();

            if (mean[itemId] > 0) {

                if (!likeSetUser.containsKey(itemId)) {
                    ArrayList<Integer> arrayList = new ArrayList<>();
                    arrayList.add(userId);
                    likeSetUser.put(itemId, arrayList);
                } else {
                    ArrayList<Integer> arrayList = new ArrayList<>();
                    arrayList.add(userId);

                    ArrayList<Integer> listTemp = likeSetUser.get(itemId);
                    arrayList.addAll(listTemp);

                    likeSetUser.put(itemId, arrayList);

                }
            }


        }

    }


    public void square() {
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
                mean[i] = sum2 / user.length;
            }
            if (sum1 > 0) {
                mapRate.put(i, Math.sqrt(sum1));
            }
        }
    }
}

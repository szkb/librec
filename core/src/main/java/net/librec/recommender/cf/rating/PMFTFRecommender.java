package net.librec.recommender.cf.rating;

import net.librec.common.LibrecException;
import net.librec.data.convertor.appender.AuxiliaryItemTagAppender;
import net.librec.data.convertor.appender.AuxiliaryTagDataAppender;
import net.librec.math.structure.MatrixEntry;
import net.librec.recommender.MatrixFactorizationRecommender;

import java.util.*;

import static net.librec.recommender.cf.rating.PMFBigItemRecommender.CAPACITY;

/**
 * @author szkb
 * @date 2019/01/19 9:47
 */
public class PMFTFRecommender extends MatrixFactorizationRecommender {


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

    HashMap<String, ArrayList<String>> itemTagInformation = new HashMap<>(CAPACITY);
    // 项目相似度
    private double[][] similarityItem;
    // 用户自身对物品评分占比多少
    private double explicitWeight = 0.8;

    private Map<Integer, String> itemIdxToItemId;
    @Override
    protected void setup() throws LibrecException {
        super.setup();
        itemTagInformation = ((AuxiliaryTagDataAppender) getDataModel().getDataAppender()).getItemTagInformation();
        itemIdxToItemId = context.getDataModel().getItemMappingData().inverse();

//        tagInformation = ((AuxiliaryItemTagAppender) getDataModel().getDataAppender()).getTagInformation();
//        maxSimilarity = new double[numUsers][numUsers];
        similarityItem = new double[numItems][numItems];
        parseItemTagInformation(itemTagInformation);
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

            String itemA = itemIdxToItemId.get(thisItem);
            int count = 0;
            for (int thatItem = thisItem + 1; thatItem < numItems; thatItem++) {

                String itemB = itemIdxToItemId.get(thatItem);


                if (!itemTagInformation.containsKey(itemA) || !itemTagInformation.containsKey(itemB)) {
                    continue;
                }

                double above = 0.0;
                double underThis = 0.0;
                double underThat = 0.0;

                ArrayList<String> thisTag = itemTagInformation.get(itemA);
                ArrayList<String> thatTag = itemTagInformation.get(itemB);

//                for (String tag : tagAmount.keySet()) {
//                    double thisTFIDF = 0;
//                    double thatTFIDF = 0;
//                    if (thisTag.contains(tag)) {
//                        thisTFIDF = tagTFIDF.get(tag);
//                    }
//                    if (thatTag.contains(tag)) {
//                        thatTFIDF = tagTFIDF.get(tag);
//                    }
//
//                    above += thisTFIDF * thatTFIDF;
//                    underThis += thisTFIDF * thisTFIDF;
//                    underThat += thatTFIDF * thatTFIDF;
//
//                }
                for (String tag : thisTag) {
                    double thisTFIDF = 0;
                    double thatTFIDF = 0;

                    thisTFIDF = tagTFIDF.get(tag);
                    underThis += thisTFIDF * thisTFIDF;

                    if (thatTag.contains(tag)) {
                        thatTFIDF = tagTFIDF.get(tag);
                        above += thisTFIDF * thatTFIDF;
                    }
                }

                for (String tag : thatTag) {
                    double thatTFIDF = tagTFIDF.get(tag);
                    underThat += thatTFIDF * thatTFIDF;
                }

                double ans = above / (Math.sqrt(underThis) * Math.sqrt(underThat));
                //</editor-fold>
                similarityItem[thisItem][thatItem] = ans;
                if (similarityItem[thisItem][thatItem] > 0) {
                    count++;
                }

            }
//            if (count > 0) {
//                System.out.println(thisItem + ":" + count);
//            }
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

    HashMap<String, Integer> tagAmount = new HashMap<>();
    HashMap<String, Double> tagTF = new HashMap<>();

    HashMap<String, Integer> tagItem = new HashMap<>();
    HashMap<String, Double> tagIDF = new HashMap<>();
    HashMap<String, Double> tagTFIDF = new HashMap<>();

    public void parseItemTagInformation(HashMap<String, ArrayList<String>> itemTagInformation) {
        for (String item : itemTagInformation.keySet()) {
            ArrayList<String> tagInformation = itemTagInformation.get(item);

            for (String tag : tagInformation) {
                if (!tagAmount.containsKey(tag)) {
                    tagAmount.put(tag, 1);
                } else {
                    int count = tagAmount.get(tag) + 1;
                    tagAmount.put(tag, count);
                }
            }

        }
        System.out.println("tagAmount:" + tagAmount.size());
        int sum = 0;
        for (String tag : tagAmount.keySet()) {
            sum += tagAmount.get(tag);
        }
        for (Map.Entry<String, Integer> entry : tagAmount.entrySet()) {
            double TFWeight = entry.getValue() * 1.0 / sum;
            tagTF.put(entry.getKey(), TFWeight);
        }

//        Iterator<Map.Entry<String, Integer>> mapIterator = tagAmount.entrySet().iterator();
//        while (mapIterator.hasNext()) {
//            Map.Entry<String, Integer> entry = mapIterator.next();
//            double IDFWeight = Math.log(numItems / (entry.getValue() + 1));
//            tagIDF.put(entry.getKey(), IDFWeight);
//        }
        for (String tag : tagAmount.keySet()) {
            for (String item : itemTagInformation.keySet()) {
                if (item.contains(tag)) {
                    if (!tagItem.containsKey(tag)) {
                        tagItem.put(tag, 1);
                    } else {
                        int count = tagItem.get(tag) + 1;
                        tagItem.put(tag, count);
                    }
                }
            }
        }

        for (String tag : tagAmount.keySet()) {
            double temp = 0;
            if (tagItem.containsKey(tag)) {
                temp = tagItem.get(tag);
            }
            double IDFWeight = Math.log(numItems / (temp + 1));
            tagIDF.put(tag, IDFWeight);
        }
        for (Map.Entry<String, Integer> entry : tagAmount.entrySet()) {
            String key = entry.getKey();
            double TFIDFWeight = tagTF.get(key) * tagIDF.get(key);
            tagTFIDF.put(key, TFIDFWeight);
        }

    }

}

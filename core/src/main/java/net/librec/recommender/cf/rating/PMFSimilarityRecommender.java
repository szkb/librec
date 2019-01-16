package net.librec.recommender.cf.rating;

import net.librec.common.LibrecException;
import net.librec.data.convertor.appender.AuxiliaryItemDataAppender;
import net.librec.math.structure.*;
import net.librec.math.structure.Vector;
import net.librec.recommender.MatrixFactorizationRecommender;
import net.librec.util.Lists;

import java.util.*;

/**
 * @author szkb
 * @date 2018/12/15 20:36
 */
public class PMFSimilarityRecommender extends MatrixFactorizationRecommender {

    private HashMap<Integer, ArrayList<Integer>> itemFeature;
    private double explicitWeight = 0.8;

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
        // todo 类中的变量不会自动初始化吗

        knn = conf.getInt("rec.neighbors.knn.number");
        similarityMatrix = context.getSimilarity().getSimilarityMatrix();
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


    }


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
        List<Map.Entry<Integer, Double>> simList = userSimilarityList[userIdx];
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

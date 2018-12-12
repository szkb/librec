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
import net.librec.math.structure.MatrixEntry;
import net.librec.math.structure.SequentialAccessSparseMatrix;
import net.librec.recommender.MatrixFactorizationRecommender;

import java.util.ArrayList;
import java.util.HashMap;

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

    private HashMap<Integer, ArrayList<Integer>> itemFeature;
    protected SequentialAccessSparseMatrix userInterestMatrix;

    @Override
    protected void setup() throws LibrecException {
        super.setup();
        itemFeature = ((AuxiliaryItemDataAppender) getDataModel().getDataAppender()).getItemFeature();

    }

    @Override
    protected void trainModel() throws LibrecException {
        for (int iter = 1; iter <= numIterations; iter++) {

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
}

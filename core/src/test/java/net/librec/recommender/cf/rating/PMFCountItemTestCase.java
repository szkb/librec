package net.librec.recommender.cf.rating;

import net.librec.BaseTestCase;
import net.librec.common.LibrecException;
import net.librec.conf.Configuration;
import net.librec.job.RecommenderJob;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;

/**
 * @author szkb
 * @date 2019/01/13 11:13
 */
public class PMFCountItemTestCase extends BaseTestCase {

    @Before
    public void setUp() throws Exception

    {
        super.setUp();
    }

    /**
     * Test the whole process of PMF Recommender
     *
     * @throws ClassNotFoundException
     * @throws LibrecException
     * @throws IOException
     */
    @Test
    public void testRecommender() throws ClassNotFoundException, LibrecException, IOException {
        Configuration.Resource resource = new Configuration.Resource("rec/cf/rating/pmfcountitem-test.properties");
        conf.addResource(resource);
        RecommenderJob job = new RecommenderJob(conf);
        job.runJob();
    }
}

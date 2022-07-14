package weka.classifiers.bayes;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import weka.classifiers.bayes.net.estimate.SimpleEstimator;
import weka.classifiers.bayes.net.search.local.K2;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;

@Slf4j
public class BayesNetInvokerTest {

    @Test
    public void testWeatherClusterByBayesNet() throws Exception {
        // 加载数据
        String file = "data/weather.nominal.arff";
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(file));
        Instances dataSet = loader.getDataSet();
        log.info("origin data set {}", dataSet);
        // 设置分类索引
        dataSet.setClassIndex(dataSet.numAttributes() - 1);
        BayesNet bayesNet = new BayesNet();
        bayesNet.setBatchSize(String.valueOf(100));
        bayesNet.setDebug(false);
        bayesNet.setDoNotCheckCapabilities(false);
        // 评估器
        SimpleEstimator estimator = new SimpleEstimator();
        estimator.setAlpha(0.5);
        bayesNet.setEstimator(estimator);
        // 搜索器
        K2 search = new K2();
        search.setMaxNrOfParents(1);
        search.setInitAsNaiveBayes(true);
        bayesNet.setSearchAlgorithm(search);
        // 使用ADTree
        bayesNet.setUseADTree(false);
        bayesNet.buildClassifier(dataSet);
    }

}

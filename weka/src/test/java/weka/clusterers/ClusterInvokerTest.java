package weka.clusterers;

import lombok.extern.slf4j.Slf4j;
import org.junit.Ignore;
import org.junit.Test;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;
import java.util.List;

/**
 * Clusterer Invoker Sample.
 *
 * @author caogaoli
 * @date 2022/7/6 19:06
 */
@Slf4j
public class ClusterInvokerTest {

    /**
     * 聚类一般步骤：
     * 1）聚类器构建：批量和增量学习；
     * 2）聚类器的评估：如果评估一个已构建的聚类器；
     * 3）聚类实例：确定未知的实例属于哪些簇；
     * <p></p>
     * 和分类器相似，构建批量聚类器也分两个阶段。
     * 第一阶段，设置参数选项；
     * 第二阶段，使用训练数据构建模型{@link Clusterer#buildClusterer(weka.core.Instances)}；
     * <p></p>
     * 聚类器评估器
     * 由于聚类是无监督的，因此很难确定一个模型到底有多好.Weka用{@link ClusterEvaluation}进行评估.
     * <p></p>
     * 提供给有监督的算法（如分类器）的数据集，也可以用于评估聚类器，由于在算法中将簇映射回类别，这种评估称为 classes-to-classes(类别的簇).
     * 类别簇的评估基本步骤：
     * 1）创建一个包含类别属性的数据集的副本，并使用过滤器{@link Remove}去除类别属性；
     * 2) 建立新的数据构建聚类器；
     * 3）使用原始数据对聚类器进行评估；
     * <p></p>
     */

    @Test
    public void testBatchCluster() throws Exception {
        // 加载数据
        String filePath = "data/contact-lenses.arff";
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(filePath));
        Instances dataSet = loader.getDataSet();
        log.info("origin dataSet {}", dataSet);
        // 构建聚类器
        String[] options = new String[2];
        // 最大迭代次数
        options[0] = "-I";
        options[1] = "100";
        // 实例化聚类器
        EM clusterer = new EM();
        // 设置选项
        clusterer.setOptions(options);
        clusterer.buildClusterer(dataSet);
        // 打印模型
        log.info("clusterer {}", clusterer);
    }

    // 构建增量聚类器
    @Test
    public void testIncrementalCluster() throws Exception {
        // 加载数据
        String filePath = "data/contact-lenses.arff";
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(filePath));
        Instances structure = loader.getStructure();
        // 训练Cobweb模型
        Cobweb cobweb = new Cobweb();
        // 先预置一个与dataSet结构类似的空Instances.
        cobweb.buildClusterer(structure);
        Instance instance;
        while ((instance = loader.getNextInstance(structure)) != null) {
            cobweb.updateClusterer(instance);
        }
        // 完成增量训练
        cobweb.updateFinished();
        // 输出模型
        log.info("clusterer Model {}", cobweb);
    }

    // 聚类器评估
    @Test
    public void testClusteringEvaluation() throws Exception {
        // 加载数据
        String file = "data/contact-lenses.arff";
        Instances data = ConverterUtils.DataSource.read(file);
        String[] options = new String[2];
        // 指定训练文件
        options[0] = "-t";
        options[1] = file;
        EM cluster = new EM();
        String output = ClusterEvaluation.evaluateClusterer(cluster, options);
        log.info("output {}", output);
        DensityBasedClusterer densityCluster = new EM();
        densityCluster.buildClusterer(data);
        ClusterEvaluation evaluation = new ClusterEvaluation();
        evaluation.setClusterer(densityCluster);
        evaluation.evaluateClusterer(new Instances(data));
        log.info("cluster result {}", evaluation.clusterResultsToString());
        // 基于密度的聚类器交叉验证
        densityCluster = new EM();
        int folds = 10;
        double logLikelyHood = ClusterEvaluation.crossValidateModel(
                densityCluster, data, folds, data.getRandomNumberGenerator(1234L));
        log.info("对数似然：{}", logLikelyHood);
    }

    // Classes-to-Clusters，评估专用于比较所选择的簇与预先设定的类别的匹配程度，在这种方式下，
    // 用户先选择一个属性（通常应该为标称型）代表"真实的" 类别。
    // 聚类数据后，weka检查每个簇中占多数的类别是哪个，并且打印混淆矩阵以显示如果使用簇来代真实类别的误差有多大.
    @Test
    public void testClassesToClusters() throws Exception {
        String file = "data/contact-lenses.arff";
        Instances data = ConverterUtils.DataSource.read(file);
        data.setClassIndex(data.numAttributes() - 1);
        // 生成聚类器数据，过滤以去除类的属性.
        Remove remove = new Remove();
        remove.setAttributeIndices("" + (data.classIndex() + 1));
        remove.setInputFormat(data);
        Instances dataCluster = Filter.useFilter(data, remove);
        log.info("origin data sample\n");
        List<Instance> origin = data.subList(0, 10);
        log.info("remove filter data sample\n");
        List<Instance> post = dataCluster.subList(0, 10);
        for (int i = 0; i < 10; ++i) {
            log.info("pre  {}", origin.get(i));
            log.info("post {}", post.get(i));
        }
        // 训练聚类器
        EM cluster = new EM();
        cluster.buildClusterer(dataCluster);
        // 评估聚类器
        ClusterEvaluation evaluation = new ClusterEvaluation();
        evaluation.setClusterer(cluster);
        evaluation.evaluateClusterer(dataCluster);
        log.info("evaluation result {}", evaluation.clusterResultsToString());
    }

    // 输出聚类分布
    @Ignore
    @Test
    public void testOutputClusterDistribution() throws Exception {
        String trainFile = "data/segment-challenge.arff";
        Instances train = ConverterUtils.DataSource.read(trainFile);
        String testFile = "data/segment-test.arff";
        Instances test = ConverterUtils.DataSource.read(testFile);
        if (!train.equalHeaders(test)) {
            throw new Exception("训练集和测试集不兼容：" + train.equalHeadersMsg(test));
        }
        // 构建聚类器
        EM clusterer = new EM();
        clusterer.buildClusterer(train);
        // 输出预测
        log.info("编号- 簇 \t-\t 分布");
        for (int i = 0; i < test.numInstances(); ++i) {
            /*int cluster = clusterer.clusterInstance(test.instance(i));
            double[] distribution = clusterer.distributionForInstance(test.instance(i));
            log.info("{}\b - \b{}\b - \b{}", i + 1, cluster, Utils.arrayToString(distribution));*/
        }
    }

}

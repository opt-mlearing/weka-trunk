package weka.filters;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Standardize;

import java.util.Random;

/**
 * filer component invoker sample.
 *
 * @author caogaoli
 * @date 2022/7/5 17:17
 */
@Slf4j
public class FilterInvokerTest {

    private static final String trainFile = "data/ReutersCorn-train.arff";
    private static final String testFile = "data/ReutersCorn-test.arff";

    // 批量过滤.
    @Test
    public void testStandardize() throws Exception {
        // 训练集
        Instances train = ConverterUtils.DataSource.read(trainFile);
        log.info("origin train Instance size {}, relation name {}",
                train.numInstances(), train.relationName());
        // 测试集
        Instances test = ConverterUtils.DataSource.read(testFile);
        log.info("origin test Instances size {}, relation name {}",
                test.numInstances(), test.relationName());
        Standardize filter = new Standardize();
        // 使用训练集初始化过滤器，规定输入数据集样例.
        filter.setInputFormat(train);
        // 使用过滤器对train和test执行过滤操作.
        Instances newTrain = Filter.useFilter(train, filter);
        Instances newTest = Filter.useFilter(test, filter);
        log.info("after filter, train Instances size {}, relation name {}",
                newTrain.numInstances(), newTrain.relationName());
        log.info("after filter, test Instances size {}, relation name {}",
                newTest.numInstances(), newTest.relationName());
    }

    // 即时数据过滤.
    @Test
    public void testFilterOnTheFly() throws Exception {
        // 训练集
        Instances train = ConverterUtils.DataSource.read(trainFile);
        train.setClassIndex(train.numAttributes() - 1);
        log.info("train info, instance size {}\b, attribute size {}\b", train.numInstances(), train.numAttributes());
        // 测试集
        Instances test = ConverterUtils.DataSource.read(testFile);
        test.setClassIndex(test.numAttributes() - 1);
        log.info("test info, instance size {}\b, attribute size {}\b", test.numInstances(), test.numAttributes());
        // 分类器
        J48 tree = new J48();
        // 使用未减裁的J48
        tree.setUnpruned(true);
        // 元分类器
        FilteredClassifier filterManager = new FilteredClassifier();
        // 过滤器
        Remove removeFilter = new Remove();
        // 删除第1个属性
        removeFilter.setAttributeIndices("1");
        // 为元分类器设置remove过滤器
        filterManager.setFilter(removeFilter);
        // 为元分类器设置分类器
        filterManager.setClassifier(tree);
        // 训练并输出模型
        filterManager.buildClassifier(train);
        log.info("model {}", filterManager);
        // 使用测试集逐个测试
        for (int i = 0; i < test.numInstances(); ++i) {
            double predicted = filterManager.classifyInstance(test.instance(i));
            log.info("编号 {}, 实际类别 {}, 预测类别 {}", i + 1,
                    test.classAttribute().value((int) test.instance(i).classValue()),
                    test.classAttribute().value((int) predicted));
        }
    }

    // 简单过滤，增加特征.
    @Test
    public void testAddFilter() throws Exception {
        String path = "data/weather.numeric.arff";
        // 加载数据集
        Instances originData = ConverterUtils.DataSource.read(path);
        log.info("origin data {}", originData);
        Instances result = new Instances(originData);
        log.info("at start, attribute size {}", result.numAttributes());
        log.info("attribute size {}", result.numAttributes());
        // 新增一个数值属性
        Add add1 = new Add();
        add1.setAttributeIndex("last");
        add1.setAttributeName("NumericAttribute");
        // 初始化add数据集样例
        add1.setInputFormat(result);
        // Filter.useFilter(result, filter)中的result是形参.
        result = Filter.useFilter(result, add1);
        log.info("after insert NumericAttribute, attribute size {}", result.numAttributes());
        // 新增一个标称属性
        Add add2 = new Add();
        add2.setAttributeIndex("last");
        // 设置标签
        add2.setNominalLabels("A,B,C");
        add2.setAttributeName("NominalAttribute");
        add2.setInputFormat(result);
        result = Filter.useFilter(result, add2);
        log.info("after insert NominalAttribute, attribute size {}", result.numAttributes());
        // 用随机值填充新增的两个属性
        Random random = new Random(1234L);
        int size = result.numAttributes();
        for (int i = 0; i < result.numInstances(); ++i) {
            result.instance(i).setValue(size - 2, random.nextDouble());
            result.instance(i).setValue(size - 1, random.nextInt(3));
        }
        log.info("finally result {}", result);
    }

    // 批量过滤
    @Test
    public void testBatchFiltering() throws Exception {
        // 加载数据
        String trainPath = "data/segment-challenge.arff";
        Instances train = ConverterUtils.DataSource.read(trainPath);
        String testPath = "data/segment-test.arff";
        Instances test = ConverterUtils.DataSource.read(testPath);
        log.info("origin train {}", train);
        log.info("origin test {}", test);
        // 过滤数据
        // 使用保准化的过程
        Standardize standardize = new Standardize();
        // 初始化过滤器的数据样式
        standardize.setInputFormat(train);
        // 基于训练集配置过滤器，并返回新的数据集
        Instances train1 = Filter.useFilter(train, standardize);
        // 基于测试集配置过滤器，并返回新的数据集
        Instances test1 = Filter.useFilter(test, standardize);
        // 打印数据集
        log.info("train {}", train1);
        log.info("test {}", test1);
    }

    // 即时过滤
    @Test
    public void testFilteringOnTheFly() throws Exception {
        // 加载数据
        String trainPath = "data/segment-challenge.arff";
        Instances train = ConverterUtils.DataSource.read(trainPath);
        String testPath = "data/segment-test.arff";
        Instances test = ConverterUtils.DataSource.read(testPath);
        // 设置类别属性
        train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes() - 1);
        // 检查训练集和测试集是否匹配
        if (!train.equalHeaders(test)) {
            throw new Exception("训练集和测试集不匹配: \n" + train.equalHeaders(test));
        }
        // 过滤器
        Remove remove = new Remove();
        remove.setAttributeIndices("1");
        // 分类器
        J48 tree = new J48();
        // 使用未减枝的的J48分类器
        tree.setUnpruned(true);
        // 元分类器
        FilteredClassifier classifier = new FilteredClassifier();
        classifier.setFilter(remove);
        classifier.setClassifier(tree);
        // 训练分类模型
        classifier.buildClassifier(train);
        // 进行预测
        for (int i = 0; i < test.numInstances(); ++i) {
            double predict = classifier.classifyInstance(test.instance(i));
            String actualClz = test.classAttribute().value((int) test.instance(i).classValue());
            String predictClz = test.classAttribute().value((int) predict);
            log.info("test instance\b {}, actual classify\b {}, J48 predicted classify\b {}",
                    i + 1, actualClz, predictClz);
        }
    }

}

package weka.filters;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Standardize;

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

}

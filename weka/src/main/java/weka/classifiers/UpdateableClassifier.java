/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    UpdateableClassifier.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers;

import weka.core.Instance;

/**
 * 增量分类器.
 * 当处理较大的的数据时，可以只占用较小的存储空间 --> 因为不必要一次性训练数据加载进内存.
 * 增量分类器主要分两个阶段完成：
 * 1）调用{@link Classifier#buildClassifier(weka.core.Instances)}进行模型初始化；
 * 2）调用{@link UpdateableClassifier#updateClassifier(weka.core.Instance)}逐行更新模型.
 * Interface to incremental classification models that can learn using one instance at a time.
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision$
 */
public interface UpdateableClassifier {

    /**
     * Updates a classifier using the given instance.
     *
     * @param instance the instance to included
     * @throws Exception if instance could not be incorporated successfully.
     */
    void updateClassifier(Instance instance) throws Exception;

}

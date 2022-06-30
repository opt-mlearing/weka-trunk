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
 *    ClassifierBeanInfo.java
 *    Copyright (C) 2002-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.gui.beans;

import java.beans.BeanDescriptor;
import java.beans.EventSetDescriptor;
import java.beans.SimpleBeanInfo;

/**
 * BeanInfo class for the Classifier wrapper bean
 *
 * @author <a href="mailto:mhall@cs.waikato.ac.nz">Mark Hall</a>
 * @version $Revision$
 */
public class ClassifierBeanInfo extends SimpleBeanInfo {

    public EventSetDescriptor[] getEventSetDescriptors() {
        try {
            EventSetDescriptor[] esds = {
                    new EventSetDescriptor(Classifier.class,
                            "batchClassifier",
                            BatchClassifierListener.class,
                            "acceptClassifier"),
                    new EventSetDescriptor(Classifier.class,
                            "graph",
                            GraphListener.class,
                            "acceptGraph"),
                    new EventSetDescriptor(Classifier.class,
                            "text",
                            TextListener.class,
                            "acceptText"),
                    new EventSetDescriptor(Classifier.class,
                            "incrementalClassifier",
                            IncrementalClassifierListener.class,
                            "acceptClassifier"),
                    new EventSetDescriptor(Classifier.class,
                            "configuration",
                            ConfigurationListener.class,
                            "acceptConfiguration")

            };
            return esds;
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        return null;
    }

    /**
     * Get the bean descriptor for this bean
     *
     * @return a <code>BeanDescriptor</code> value
     */
    public BeanDescriptor getBeanDescriptor() {
        return new BeanDescriptor(weka.gui.beans.Classifier.class,
                ClassifierCustomizer.class);
    }
}


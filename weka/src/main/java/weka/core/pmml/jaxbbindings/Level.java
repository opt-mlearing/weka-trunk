//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.0-b52-fcs 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2013.12.20 at 12:48:21 PM GMT 
//


package weka.core.pmml.jaxbbindings;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for Level element declaration.
 *
 * <p>The following schema fragment specifies the expected content contained within this class.
 *
 * <pre>
 * &lt;element name="Level">
 *   &lt;complexType>
 *     &lt;complexContent>
 *       &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *         &lt;attribute name="alpha" type="{http://www.dmg.org/PMML-4_1}REAL-NUMBER" />
 *         &lt;attribute name="smoothedValue" type="{http://www.dmg.org/PMML-4_1}REAL-NUMBER" />
 *       &lt;/restriction>
 *     &lt;/complexContent>
 *   &lt;/complexType>
 * &lt;/element>
 * </pre>
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "")
@XmlRootElement(name = "Level")
public class Level {

    @XmlAttribute
    protected Double alpha;
    @XmlAttribute
    protected Double smoothedValue;

    /**
     * Gets the value of the alpha property.
     *
     * @return possible object is
     * {@link Double }
     */
    public Double getAlpha() {
        return alpha;
    }

    /**
     * Sets the value of the alpha property.
     *
     * @param value allowed object is
     *              {@link Double }
     */
    public void setAlpha(Double value) {
        this.alpha = value;
    }

    /**
     * Gets the value of the smoothedValue property.
     *
     * @return possible object is
     * {@link Double }
     */
    public Double getSmoothedValue() {
        return smoothedValue;
    }

    /**
     * Sets the value of the smoothedValue property.
     *
     * @param value allowed object is
     *              {@link Double }
     */
    public void setSmoothedValue(Double value) {
        this.smoothedValue = value;
    }

}

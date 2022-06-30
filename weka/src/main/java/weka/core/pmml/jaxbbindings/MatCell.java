//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.0-b52-fcs 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2013.12.20 at 12:48:21 PM GMT 
//


package weka.core.pmml.jaxbbindings;

import java.math.BigInteger;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.XmlType;
import javax.xml.bind.annotation.XmlValue;


/**
 * <p>Java class for MatCell element declaration.
 *
 * <p>The following schema fragment specifies the expected content contained within this class.
 *
 * <pre>
 * &lt;element name="MatCell">
 *   &lt;complexType>
 *     &lt;simpleContent>
 *       &lt;extension base="&lt;http://www.w3.org/2001/XMLSchema>string">
 *         &lt;attribute name="col" use="required" type="{http://www.dmg.org/PMML-4_1}INT-NUMBER" />
 *         &lt;attribute name="row" use="required" type="{http://www.dmg.org/PMML-4_1}INT-NUMBER" />
 *       &lt;/extension>
 *     &lt;/simpleContent>
 *   &lt;/complexType>
 * &lt;/element>
 * </pre>
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "", propOrder = {
        "value"
})
@XmlRootElement(name = "MatCell")
public class MatCell {

    @XmlValue
    protected String value;
    @XmlAttribute(required = true)
    protected BigInteger col;
    @XmlAttribute(required = true)
    protected BigInteger row;

    /**
     * Gets the value of the value property.
     *
     * @return possible object is
     * {@link String }
     */
    public String getValue() {
        return value;
    }

    /**
     * Sets the value of the value property.
     *
     * @param value allowed object is
     *              {@link String }
     */
    public void setValue(String value) {
        this.value = value;
    }

    /**
     * Gets the value of the col property.
     *
     * @return possible object is
     * {@link BigInteger }
     */
    public BigInteger getCol() {
        return col;
    }

    /**
     * Sets the value of the col property.
     *
     * @param value allowed object is
     *              {@link BigInteger }
     */
    public void setCol(BigInteger value) {
        this.col = value;
    }

    /**
     * Gets the value of the row property.
     *
     * @return possible object is
     * {@link BigInteger }
     */
    public BigInteger getRow() {
        return row;
    }

    /**
     * Sets the value of the row property.
     *
     * @param value allowed object is
     *              {@link BigInteger }
     */
    public void setRow(BigInteger value) {
        this.row = value;
    }

}

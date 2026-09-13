
# ============================================================
# SuperKart Sales Prediction - Streamlit Application
# ============================================================

# ------------------------------------------------------------
# Import required libraries
# ------------------------------------------------------------
import os
import streamlit as st
import pandas as pd
import joblib


# ============================================================
# 1. Page Configuration
# ============================================================

st.set_page_config(
    page_title="SuperKart Sales Prediction",
    page_icon="🛒",
    layout="centered"
)


# ============================================================
# 2. Load Trained Model
# ============================================================

model_path = os.path.join(
    os.path.dirname(__file__),
    "best_superkart_sales_model_v1.joblib"
)


if not os.path.exists(model_path):

    st.error(
        "Trained model file was not found. "
        "Please run the model training pipeline before "
        "starting the application."
    )

    st.code(model_path)

    st.stop()


try:

    model = joblib.load(model_path)

except Exception as error:

    st.error(
        f"Unable to load the trained model: {error}"
    )

    st.stop()


# ============================================================
# 3. Application Title
# ============================================================

st.title("🛒 SuperKart Sales Prediction App")

st.write(
    """
    This application predicts the estimated total sales of a
    product at a SuperKart store based on product and store
    characteristics.

    Enter the required details below and click
    **Predict Sales** to generate the estimated sales value.
    """
)


# ============================================================
# 4. Product Details
# ============================================================

st.subheader("📦 Product Details")


product_weight = st.number_input(
    label="Product Weight",
    min_value=0.0,
    max_value=100.0,
    value=12.50,
    step=0.01
)


product_sugar_content = st.selectbox(
    label="Product Sugar Content",
    options=[
        "Low Sugar",
        "Regular",
        "No Sugar"
    ]
)


product_allocated_area = st.number_input(
    label="Product Allocated Area",
    min_value=0.0,
    max_value=1.0,
    value=0.05,
    step=0.001,
    format="%.3f"
)


product_type = st.selectbox(
    label="Product Type",
    options=[
        "Baking Goods",
        "Breads",
        "Breakfast",
        "Canned",
        "Dairy",
        "Frozen Foods",
        "Fruits and Vegetables",
        "Hard Drinks",
        "Health and Hygiene",
        "Household",
        "Meat",
        "Others",
        "Seafood",
        "Snack Foods",
        "Soft Drinks",
        "Starchy Foods"
    ]
)


product_mrp = st.number_input(
    label="Product MRP",
    min_value=0.0,
    max_value=1000.0,
    value=150.00,
    step=0.01
)


# ============================================================
# 5. Store Details
# ============================================================

st.subheader("🏪 Store Details")


store_establishment_year = st.number_input(
    label="Store Establishment Year",
    min_value=1900,
    max_value=2100,
    value=2009,
    step=1
)


store_size = st.selectbox(
    label="Store Size",
    options=[
        "Small",
        "Medium",
        "High"
    ]
)


store_location_city_type = st.selectbox(
    label="Store Location City Type",
    options=[
        "Tier 1",
        "Tier 2",
        "Tier 3"
    ]
)


store_type = st.selectbox(
    label="Store Type",
    options=[
        "Departmental Store",
        "Food Mart",
        "Supermarket Type1",
        "Supermarket Type2"
    ]
)


# ============================================================
# 6. Create Model Input
# ============================================================

input_data = pd.DataFrame(
    [{
        "Product_Weight": product_weight,

        "Product_Sugar_Content":
            product_sugar_content,

        "Product_Allocated_Area":
            product_allocated_area,

        "Product_Type":
            product_type,

        "Product_MRP":
            product_mrp,

        "Store_Establishment_Year":
            store_establishment_year,

        "Store_Size":
            store_size,

        "Store_Location_City_Type":
            store_location_city_type,

        "Store_Type":
            store_type
    }]
)


# ============================================================
# 7. Display Input Summary
# ============================================================

with st.expander("View Input Details"):

    st.dataframe(
        input_data,
        use_container_width=True
    )


# ============================================================
# 8. Prediction Button
# ============================================================

if st.button(
    "🔮 Predict Sales",
    type="primary",
    use_container_width=True
):

    try:

        # ----------------------------------------------------
        # Generate prediction
        # ----------------------------------------------------

        prediction = model.predict(
            input_data
        )[0]


        # ----------------------------------------------------
        # Display prediction
        # ----------------------------------------------------

        st.success(
            "Sales prediction generated successfully!"
        )


        st.metric(
            label="Estimated Product Store Sales",
            value=f"{prediction:,.2f}"
        )


        # ----------------------------------------------------
        # Additional information
        # ----------------------------------------------------

        st.info(
            f"""
            The estimated total sales for the selected
            product and store configuration is:

            **{prediction:,.2f}**
            """
        )


    except Exception as error:

        st.error(
            f"Prediction failed: {error}"
        )


# ============================================================
# 9. Application Footer
# ============================================================

st.markdown("---")

st.caption(
    "SuperKart Sales Prediction | "
    "XGBoost Regression Model"
)

import streamlit as st
from streamlit_extras.colored_header import colored_header

# just a landing page with ChatGPT generated text.
def intro():
    st.image("MealOMeterCrop.png")

    colored_header(
            label="",
            description="Made by People of Science",
            color_name="red-70",
        )
    st.write("### The Future Is Here!")
    st.markdown("""
        Welcome to the future of meal planning, folks! Say hello to **Meal-O-Meter**, the smartest way to bring nutrition, convenience, and delightful meals right to your kitchen table. Created by the bright minds of modern science, Meal-O-Meter is here to make meal planning a breeze—just like those futuristic kitchens they promised us!

        #### Designed by Doctors, for Your Family
        The Meal-O-Meter is the result of tireless work by the best and brightest in food science, nutrition, and technology. Our experts put their slide rules together to craft a meal-planning experience that's as easy as pie. It takes the guesswork out of nutrition and puts the joy back into cooking. Think of it as having your very own kitchen assistant, ready to create perfectly balanced meals for the whole family, day in and day out!

        #### Endorsed by Dr. Cal O'Ries
        Yes, that's right—**Dr. Cal O'Ries**, the nation’s favorite nutrition expert, recommends the Meal-O-Meter! "The Meal-O-Meter," he says, "is a triumph of modern ingenuity! With its carefully balanced meal suggestions and its ability to cater to every family’s taste, it makes eating healthy fun and effortless."

""")
                
    col1,col2,col3 = st.columns(3)
    with col2:
        st.header("", divider="grey")


    st.markdown("""
        #### Join the Future!
        Forget the old days of complicated recipes and last-minute trips to the grocery store. The Meal-O-Meter puts the power of planning in your hands—simplifying your grocery list, optimizing your meal schedule, and making sure your family gets wholesome, nutritious meals every day. It's meal planning, the way it should be: **smart, simple, and oh-so-delicious**.

        Step into tomorrow, today, with **Meal-O-Meter**. Because when science and nutrition come together, the future looks delicious!

        """)


intro()
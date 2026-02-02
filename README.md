# Set solver
This is a computer vision python project which solves the card game [Set](https://en.wikipedia.org/wiki/Set_(card_game)): You upload an image and then see which sets are in it.

## Usage
1. Navigate to https://solve-set-ck.streamlit.app
   - If no one has accessed the URL for a while, you'll need to "wake up" the app. It should only take a couple of seconds.
2. Upload an image (with cards from Set).
   - For best results, take the photo in a well-lit environment while avoiding glare. Lighting conditions are fortunately not that strict, though---image quality is more important, but any modern phone camera provides plenty of resolution.
3. Wait a few seconds and view the sets![^1]

## Example[^2]
Here's an image with some sets:

<img src='tests/test.jpg' width='800'>

### Identifying features
Every card has four different features, each of which have three varieties. 

| color   |  shape     | number | filling |
| ------- | ---------- | ------ | ------- |
| purple  |  squiggle  |   1    |  empty   | 
| red     |   rhombus  |   2    |  dashed  | 
| green   |   oval     |   3    |  full    |

The card features as identified by the program are pictured below; each feature is abbreviated by its first or first two letters ('P' is for 'purple', 'OV' is for 'oval', 'E' is for empty, and so on). You can check for yourself that they are correct.

<img src='https://github.com/user-attachments/assets/1feebc33-d118-48cb-a4f9-630eba8bb7cf' width='800'>

### Sets
Finally, here are all 5 sets highlited:

<img src='https://github.com/user-attachments/assets/20644c79-7afe-46ca-99c8-a9fb094bc9ea' width='325'>
<img src='https://github.com/user-attachments/assets/a40280eb-7cc7-4d3f-b798-76efb6a1dcbb' width='325'>
<img src='https://github.com/user-attachments/assets/8a7f1fd8-14fc-467b-bd13-2f927bbd9d51' width='325'>

<img src='https://github.com/user-attachments/assets/8ac4810f-4538-4647-a48a-6edd58aabdde' width='325'>
<img src='https://github.com/user-attachments/assets/c03d95dc-9b07-4b71-af08-a868d1777af1' width='325'>

[^1]: I'm happy with this program's success rate---it has identified cards correctly each time I've used it---but there's definitely room for improvement. If the image resolution is low, lighting is poor, cards are touching or obscurred, or glare is prevalent, the cards will likely not be identified correctly (and so neither will the sets).

[^2]: You can also the web app's printed-page output for this example at [tests/exampleoutput.pdf](https://github.com/charleskolozsvary/solve-set/blob/main/tests/exampleoutput.pdf).



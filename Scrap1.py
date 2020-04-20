from bs4 import BeautifulSoup
import requests

with open('Friends.html') as html_file:
    soup = BeautifulSoup(html_file,'lxml')

find = soup.find('div',class_='spch s2fp-h')
print(find)

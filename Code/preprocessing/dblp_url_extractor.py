import mysql.connector

from mysql.connector import errorcode


def get_mysql_db(host='127.0.0.1'):
    try:
        return mysql.connector.connect(host=host, user='root')
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        else:
            print(err)


if __name__ == '__main__':
    conn = get_mysql_db()
    c = conn.cursor()
    c.execute('USE airankings')
    sql = 'select a.name, au.url from author a natural join author_url au ' \
          'natural join author_country ac where ac.country = %s;'
    p = ('Germany',)
    c.execute(sql, p)
    authors = c.fetchall()
    authors_dict = {}
    for auth in authors:
        name, url = auth
        if name in authors_dict:
            authors_dict[name].append(url)
        else:
            authors_dict[name] = [url]
    with open("./data/dblp_name_url.json", 'w', encoding='utf-8') as f:
        for name, urls in authors_dict.items():
            print(name, ":", urls)
            f.write(name + ";" + ",".join(urls) + "\n")
    print("Finished. ")

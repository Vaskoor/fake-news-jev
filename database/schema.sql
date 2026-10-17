
CREATE TABLE articles (id SERIAL PRIMARY KEY, headline TEXT NOT NULL, body TEXT, source TEXT, url TEXT, date DATE, label TEXT CHECK (label IN ('REAL','FAKE','NEEDS_REVIEW')), probability REAL, created_at TIMESTAMP DEFAULT NOW());
CREATE TABLE signals (id SERIAL PRIMARY KEY, article_id INT REFERENCES articles(id) ON DELETE CASCADE, name TEXT NOT NULL, type TEXT NOT NULL, value REAL, probability REAL, contribution REAL);
CREATE TABLE swipes (id SERIAL PRIMARY KEY, article_id INT REFERENCES articles(id), action TEXT CHECK (action IN ('accept','reject')), created_at TIMESTAMP DEFAULT NOW());
CREATE TABLE feedback (id SERIAL PRIMARY KEY, article_id INT REFERENCES articles(id), user_label TEXT, notes TEXT, created_at TIMESTAMP DEFAULT NOW());
CREATE TABLE model_runs (id SERIAL PRIMARY KEY, model_name TEXT, accuracy REAL, f1 REAL, ece REAL, config TEXT, created_at TIMESTAMP DEFAULT NOW());
CREATE INDEX idx_signals_article ON signals(article_id);

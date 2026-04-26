# tutorial-linear-regression-simulator-app

> Bilingual README (Persian + English)  
> [Jump to English](#english-version)

[![GitHub Repo](https://img.shields.io/badge/Repository-khalooei%2Ftutorial--linear--regression--simulator--app-181717?logo=github)](https://github.com/khalooei/tutorial-linear-regression-simulator-app)
[![Star this project](https://img.shields.io/badge/If%20you%20liked%20it-Give%20a%20Star-fbbf24)](https://github.com/khalooei/tutorial-linear-regression-simulator-app/stargazers)

<div dir="rtl">

## نسخه فارسی (RTL)

### معرفی
این پروژه یک ابزار شبیه سازی آموزشی و تعاملی برای درس **بهینه سازی** است که با یک مثال رگرسیون خطی، مفاهیم زیر را شفاف نشان می دهد:

- تفاوت **فضای نمونه** (`x, y`) و **فضای پارامتر** (`w, b`)
- تاثیر تغییر پارامترها روی **خط برازش**
- تغییر مقدار **Loss** و مسیر حرکت **Gradient Descent**
- مشاهده جزئیات فرمولی هر گام با نمایش **LaTeX**

### پیش نمایش ویدئویی
<video src="video/overview.webm" controls muted playsinline width="100%"></video>

### قابلیت ها
- نمایش همزمان فضای نمونه و فضای پارامتر
- انتخاب تابع هزینه: `MSE`، `MAE`، `Huber`
- انیمیشن GD با:
  - نرخ یادگیری قابل تنظیم
  - تعداد گام قابل تنظیم
  - تاخیر قابل تنظیم
  - امکان **Stop** در حین اجرا
- سایدبار راست برای گزارش مرحله به مرحله:
  - پارامترهای فعلی
  - گرادیان
  - فرمول و مقدار Loss
  - قانون آپدیت پارامتر
- رابط کاربری دو زبانه (فارسی/انگلیسی)
- تم **شب / روز**

### اجرای سریع
1) کلون پروژه:

```bash
git clone git@github.com:khalooei/tutorial-linear-regression-simulator-app.git
cd tutorial-linear-regression-simulator-app
```

2) ساخت محیط مجازی:

```bash
python -m venv .venv
```

3) فعال سازی محیط:

Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:
```bash
source .venv/bin/activate
```

4) نصب وابستگی ها:

```bash
pip install -r requirements.txt
```

5) اجرای برنامه:

```bash
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8765
```

6) آدرس اجرا:

`http://127.0.0.1:8765`

### سناریوی تدریس (درس بهینه سازی)
1. تفاوت فضای نمونه و فضای پارامتر را معرفی کنید.
2. اسلایدرهای `w` و `b` را تغییر دهید و اثر آن را روی خط و خطا نشان دهید.
3. تابع Loss را بین `MSE`/`MAE`/`Huber` تغییر دهید.
4. انیمیشن GD را اجرا کنید و مسیر حرکت روی سطح خطا را بررسی کنید.
5. پنل جزئیات را باز کنید و محاسبات هر مرحله را خط به خط مرور کنید.

### اگر پروژه را دوست داشتید
از طریق لینک زیر به پروژه **Star** بدهید:

👉 [Star on GitHub](https://github.com/khalooei/tutorial-linear-regression-simulator-app/stargazers)

### ارتباط و همکاری توسعه
اگر سوالی داشتید یا برای توسعه و همکاری آموزشی/پژوهشی علاقه مند بودید، خوشحال می شوم در ارتباط باشیم.

**Email:** `khalooei@aut.ac.ir`

### نویسنده
**mohammad khalooei**

</div>

---

## English Version

### Overview
This project is an interactive teaching simulator for **Optimization** courses using a linear regression scenario.  
It clearly demonstrates:

- the difference between **sample space** (`x, y`) and **parameter space** (`w, b`)
- how parameter changes affect the fitted line
- how **loss** changes and how **gradient descent** moves
- step-by-step mathematical details using **LaTeX**

### Video Preview
<video src="video/overview.webm" controls muted playsinline width="100%"></video>

### Features
- Side-by-side sample space and parameter space views
- Selectable losses: `MSE`, `MAE`, `Huber`
- Gradient descent animation with configurable:
  - learning rate
  - number of steps
  - delay
  - real-time **Stop** button
- Right details sidebar with per-step:
  - current parameters
  - gradients
  - loss formula and value
  - update equation
- Bilingual interface (Persian / English)
- Dark / Light theme

### Quick Start
1) Clone:

```bash
git clone git@github.com:khalooei/tutorial-linear-regression-simulator-app.git
cd tutorial-linear-regression-simulator-app
```

2) Create venv:

```bash
python -m venv .venv
```

3) Activate:

Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:
```bash
source .venv/bin/activate
```

4) Install requirements:

```bash
pip install -r requirements.txt
```

5) Run:

```bash
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8765
```

6) Open:

`http://127.0.0.1:8765`

### If You Like This Project
Please support it with a GitHub star:

👉 [Star this repository](https://github.com/khalooei/tutorial-linear-regression-simulator-app/stargazers)

### Contact & Collaboration
For questions, educational usage, or development collaboration:

**Email:** `khalooei@aut.ac.ir`

### Author
**mohammad khalooei**


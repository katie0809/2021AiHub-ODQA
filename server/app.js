const express = require("express");
const cors = require("cors");
const dotenv = require("dotenv");
const path = require("path");
const mode = process.env.MODE;

dotenv.config({
  path: path.resolve(`.env.${mode}`),
});

/** **************************************************** */
// 모듈 초기화
const logger = require("../Plugin/Logger");
const app = express();

/** CORS 허용: 아이폰에 대해서만 허용한다. */
const corsOptions = {
  origin: "ionic://localhost",
  optionsSuccessStatus: 200,
};
app.use(cors(corsOptions));

/** 전역변수 세팅 */
app.locals.isDev = process.env.NODE_ENV === "development";

/** 포트 세팅 */
app.set("port", process.env.PORT);

/** body-parser 사용 */
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

/** API 로그생성 */
app.use(logger.express);

/** 라우터 */
const chatrRouter = require("../services/chat");

/** 에러처리 */
app.use((err, req, res, next) => {
  const errStatus = err.status || err.statusCode;
  const errCode = err.code || "000";
  const errMessage = err.msg || err.message || err.stack;

  logger.error(err, req);

  if (!errStatus) {
    res.status(500).json({
      errCode,
      errMessage,
    });
  } else {
    res.status(errStatus).json({
      errCode,
      errMessage,
    });
  }
});

const server = app.listen(process.env.PORT, function () {
  console.log(`[${process.pid}] Express server has started at ${process.env.HOST} on port ${process.env.PORT}`);
});

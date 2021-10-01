const log4js = require("log4js");

/** 로깅 환경설정 */
log4js.configure({
  appenders: {
    default: {
      type: "dateFile",
      filename: "logs/system.log",
      maxLogSize: 10485760,
      pattern: ".yyyy-MM-dd",
      keepFileExt: true,
      compress: false,
      pm2: true,
      layout: {
        type: "pattern",
        pattern: "[%d] [%p] %m",
      },
    }
  },
  categories: {
    default: {
      appenders: ["default"],
      level: "trace",
    }
  },
});
const writer = log4js.getLogger("default");

/** 로그 메시지로 남길 메시지 객체를 String 형식으로 변환한다. */
const stringifyPayload = (PAYLOAD) => {
  switch (typeof PAYLOAD) {
    case "object":
      return JSON.stringify(PAYLOAD);
    case "string":
      return PAYLOAD;
    default:
      return PAYLOAD.toString();
  }
};

module.exports = {
  express: log4js.connectLogger(writer, { level: log4js.levels.TRACE }),
  message: (msg) => {
    if (process.env.MODE === "dev") {
      // write custome console log only for dev mode
      console.log(msg);
      writer.info(`MESSAGE :: ${msg}`);
    }
  },
  /**
   * header : 요청 혹은 응답을 의미하는 헤더. REQUEST/RESPONSE의 스트링 문자열
   * message : 로그로 남길 메시지, 혹은 객체 값.
   * path : 요청 혹은 응답 url
   */
  debug: (header, message, path = null) => {
    try {
      const HEADER = header;
      const PAYLOAD = message;

      if (HEADER === "REQUEST" || HEADER === "RESPONSE") {
        const URL = path;

        writer.debug(`${HEADER} :: url=${URL} payload=${stringifyPayload(PAYLOAD)}`);
        console.log(`${HEADER} :: url=${URL} payload=${stringifyPayload(PAYLOAD)}`);
      } else {
        writer.debug(`${HEADER} :: ${PAYLOAD}`);
        console.log(`${HEADER} :: ${PAYLOAD}`);
      }
    } catch (e) {
      writer.error(e);
    }
  },
  warn: (header, message, path = null) => {
    try {
      const HEADER = header;
      const PAYLOAD = message;

      if (HEADER === "REQUEST" || HEADER === "RESPONSE") {
        const URL = path;

        writer.warn(`${HEADER} :: url=${URL} payload=${stringifyPayload(PAYLOAD)}`);
      } else {
        writer.warn(`${HEADER} :: ${PAYLOAD}`);
      }
    } catch (e) {
      writer.error(e);
    }
  },
  error: (err, req = null) => {
    if (!err) return;
    if (!req) {
      writer.error(err.stack);
    } else {
      const URL = req.originalUrl;
      let errCode = err.status || err.statusCode;
      let errMessage = err.message;

      if (err.code) errCode = `${errCode}(${err.code})`;
      if (err.msg) errMessage = `client_msg=${err.msg} message=${errMessage}`;

      writer.error(`${URL} :: error=[${errCode}] ${errMessage}\n${err.stack}`);
    }
  }
};

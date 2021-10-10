const express = require("express");
const createError = require("http-errors");
const base64url = require("base64url");

const strings = require("../config/strings");

const logger = require("../Plugin/Logger");
const ChatHandler = require("../Plugin/Chat");
const router = express.Router();

/** 질의 답변 요청 */
router.post("/", (req, res, next) => {
  try {
    // 요청객체 유효성 체크
    if (!req || !res || !req.body || !req.body.message) {
      next(createError(405, strings.err_wrong_params("message")));
      return;
    }

    // 요청객체 값 추출
    const { message } = req.body;
    // chat.checkParamValid(req.body);
    logger.debug("REQUEST", req.body, req.originalUrl);

    const successCallback = (answer) => {

      const resBody = {
        message: message, // 기존 메시지
        answer: answer // 챗봇 답변
      };

      res.status(200).json(resBody);
      logger.debug("RESPONSE", resBody, req.originalUrl);
    };
    const errorCallback = (error) => {
      next(createError(520, strings.err_chat_fail));
    };
    const chat = new ChatHandler(message, successCallback, errorCallback);


  } catch (e) {
    next(e);
  }
});

/** 질의 토큰화 결과 요청 */
router.post("/tokens", (req, res, next) => {
    try {

    } catch(e) {
        next(e);
    }
})
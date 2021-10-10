const createError = require("http-errors");
const strings = require("../config/strings");
const logger = require("./Logger");

class Chat {
  constructor(message, successCallbark, errCallback) {
    if (!message || successCallbark === null || errCallback === undefined) {
      throw createError(405, strings.err_wrong_params);
    }

    /** 단말 -> 서버 전송필수값(키: 길이) */
    this.requiredClientParams = { message: 1024 };
  }

  /**
   * 요청 파라미터에 대한 타입과 길이 유효성 체크한다
   * @param {Object} param 요청 파라미터
   */
  checkParamValid(param) {
    Object.entries(this.requiredClientParams).map(([key, value]) => {
      if (!Object.prototype.hasOwnProperty.call(param, key)) throw createError(405, strings.err_no_params(key));
      else if (param[key].length > value) throw createError(405, strings.err_wrong_params(key));

      return true;
    });
  }
}

module.exports = Chat;

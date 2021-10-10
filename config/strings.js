const compInfo = require("./companies");

module.exports = {
  /** 요청객체 오류 */
  err_no_params: (key=null) => {
    if(key==null) '요청정보가 잘못되었습니다.'
    else return `요청정보가 잘못되었습니다. 오류([${key}] 누락)`;
  },
  err_wrong_params: (key) => {
    return `요청정보가 잘못되었습니다. 오류([${key}])`;
  },

  /** 기타 오류 */
  err_chat_fail: "다시 시도해주시기 바랍니다. 반복하여 문제가 생길경우 문의바랍니다."
};

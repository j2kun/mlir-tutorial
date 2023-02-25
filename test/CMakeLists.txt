configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TUTORIAL_TEST_DEPENDS
        FileCheck count not
        # tutorial-opt
        # tutorial-translate
        )

add_lit_testsuite(check-tutorial "Running tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${TUTORIAL_TEST_DEPENDS}
        )
set_target_properties(check-tutorial PROPERTIES FOLDER "Tests")

add_lit_testsuites(TUTORIAL ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TUTORIAL_TEST_DEPENDS})
